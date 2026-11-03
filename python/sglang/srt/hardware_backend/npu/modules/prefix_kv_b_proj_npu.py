"""Merge cache-hit prefix and current-token ``kv_b_proj`` for NPU MHA prefill.

The eager GLM-4.7-Flash MHA prefill path projects KV-B twice per layer:

1. ``forward_mha_prepare_npu`` projects the current tokens with the statically
   split ``kv_b_proj`` (``SGLANG_NPU_SPLIT_KV_B_PROJ``): two GEMMs on
   ``[new_tokens, 512]`` producing K_nope and V.
2. ``AscendAttnBackend.forward_extend``'s prefix branch gathers the cached
   historical latent, runs the ORIGINAL packed ``kv_b_proj`` on the prefix
   rows (a third GEMM whose packed width is ``H*(Dk+Dv)``), then concatenates
   history/current K and V per request before FIA.

This module merges the input rows instead of the outputs: the historical
latent chunk of every request ``[0:p_j]`` and its current chunk ``[0:q_j]``
are packed in request order ``[p0, c0, p1, c1, ...]`` into one contiguous
``[sum(p+q), 512]`` tensor (and the already-rotated K-RoPE rows in the same
order).  The existing split projection then runs exactly once on the merged
rows, so the prefix-only packed GEMM and the separate current-only GEMMs are
replaced by exactly two GEMMs at ``M = sum(prefix + new)``.  Outputs are
already in FIA request order, so the backend consumes contiguous views and
the historical/current sequence-axis K/V concatenations disappear.

Narrow opt-in (default off), requiring ALL of:

  * ``SGLANG_NPU_MERGE_PREFIX_KV_B_PROJ=1``;
  * the plain ``EXTEND`` forward mode on the eager Ascend backend, with the
    unified writer having produced the current normalized latent and rotated
    K-RoPE (``SGLANG_NPU_KV_RMSNORM_UNIFIED_CACHE``), the page-first prefix
    gather gate enabled, and positive prefix pages;
  * unified 576-wide latent pool: latent512 / rope64 / page128, BF16 or FP16
    (store dtype == dtype), no DSA / PA_NZ / expanded-KV geometry;
  * the per-head split of ``kv_b_proj`` already prepared and current (the
    existing stale-weight guard is reused, never weakened), TP=1;
  * NPU GLM geometry ``qk_nope=192 / rope=64 / v=256``, equal local q/k/v
    heads, no cross-attention / sliding window / CP / MLAPO / sinks /
    speculative path / graph capture / torch.compile.

Anything else returns ``None`` before any contract changes, so the original
prepare and backend routes stay responsible and run verbatim.  Device/runtime
errors are not caught.

The host query/KV intervals are carried in :class:`MergedPrefixKVProjection`,
a per-call value returned from prepare to the attention core.  Nothing is
cached on the attention module, backend or ``ForwardBatch``.  The dedicated
backend consumer slices ``K_full[kv_start:kv_end]`` / ``V_full[...]`` and the
new-token query rows only; it reuses the original FIA per-request call and the
existing direct-output helper, so the numeric attention algorithm, sparse
causal mode and output layout are unchanged.
"""

from __future__ import annotations

import functools
import numbers
import operator
import os
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch

_ENV = "SGLANG_NPU_MERGE_PREFIX_KV_B_PROJ"

_KEY_LATENT = 512
_KEY_ROPE = 64
_PAGE = 128
_KEY_NOPE = 192
_VALUE_DIM = 256
# Attention Q/K head width is K_nope + K_rope (192 + 64).  The packed KV-B
# output width H*(K_nope + V) (448 per head) is a different quantity and must
# never be used for the FIA/Q head dimension.
_QK_HEAD_DIM = _KEY_NOPE + _KEY_ROPE

_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16)


# ---------------------------------------------------------------------------
# gate
# ---------------------------------------------------------------------------
def _env_enabled() -> bool:
    try:
        from sglang.srt.utils import get_bool_env_var

        return get_bool_env_var(_ENV, "false")
    except Exception:
        return os.environ.get(_ENV, "0").strip().lower() in ("1", "true", "yes", "on")


@functools.lru_cache(maxsize=1)
def merge_prefix_kv_b_proj_enabled() -> bool:
    """Cached opt-in gate.  Never parsed per layer."""
    return _env_enabled()


def clear_gate_cache() -> None:
    """Test hook: re-read the environment switch."""
    merge_prefix_kv_b_proj_enabled.cache_clear()


def _graph_or_compile_active() -> bool:
    try:
        if torch.compiler.is_compiling():
            return True
    except Exception:
        return True
    npu = getattr(torch, "npu", None)
    capture = (
        getattr(npu, "is_current_stream_capturing", None) if npu is not None else None
    )
    if capture is not None:
        try:
            if capture():
                return True
        except Exception:
            return True
    return False


def _backend_is_supported(backend: object) -> bool:
    """Only the actual eager Ascend attention backend is handled.

    TBO wrappers and any alternate backend are rejected; the ordinary prepare
    and backend routes stay responsible for them.
    """
    try:
        from sglang.srt.hardware_backend.npu.attention.ascend_backend import (
            AscendAttnBackend,
        )
    except Exception:
        return False
    return isinstance(backend, AscendAttnBackend)


# ---------------------------------------------------------------------------
# host metadata / layout
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PrefixMergeLayout:
    """Per-request host intervals for the merged ``[prefix, current]`` rows.

    ``kv_offsets[j]`` is the row where request ``j``'s ``[p_j, q_j]`` block
    starts in the merged latent/K/V tensors; ``q_offsets[j]`` is the first
    new-token row of request ``j`` in the original ``q`` tensor.
    """

    prefix_lens: Tuple[int, ...]
    seq_lens: Tuple[int, ...]
    pages_per_req: Tuple[int, ...]
    page_offsets: Tuple[int, ...]
    kv_offsets: Tuple[int, ...]
    q_offsets: Tuple[int, ...]
    total_rows: int
    total_pages: int
    sum_prefix: int
    sum_seq: int


def _as_host_int(value: object) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        return None
    return operator.index(value)


def build_merge_layout(
    prefix_lens: Optional[Sequence[int]],
    seq_lens: Optional[Sequence[int]],
    num_selected_pages: int,
    *,
    page_size: int = _PAGE,
) -> Optional[PrefixMergeLayout]:
    """Validate the existing HOST length lists and derive the merge offsets.

    Pure Python (no tensor reads).  ``num_selected_pages`` is the host
    ``flatten_prefix_block_tables.numel()`` produced by the backend, so the
    page padding implied by per-request ``ceil(p_j / page_size)`` pages is
    checked here against the actual gather metadata without a device sync.
    """
    if prefix_lens is None or seq_lens is None:
        return None
    if len(prefix_lens) == 0 or len(prefix_lens) != len(seq_lens):
        return None
    pages_total = _as_host_int(num_selected_pages)
    if pages_total is None or pages_total < 0:
        return None

    ps: list[int] = []
    qs: list[int] = []
    for raw_p, raw_q in zip(prefix_lens, seq_lens):
        p = _as_host_int(raw_p)
        q = _as_host_int(raw_q)
        if p is None or q is None or p < 0 or q < 0:
            return None
        ps.append(p)
        qs.append(q)

    sum_p = sum(ps)
    sum_q = sum(qs)
    if sum_p <= 0 or sum_q <= 0:
        return None

    pages_per = tuple((p + page_size - 1) // page_size for p in ps)
    if sum(pages_per) != pages_total:
        return None

    page_offsets: list[int] = []
    kv_offsets: list[int] = []
    q_offsets: list[int] = []
    page_cursor = 0
    row_cursor = 0
    q_cursor = 0
    for p, q, pages in zip(ps, qs, pages_per):
        page_offsets.append(page_cursor)
        kv_offsets.append(row_cursor)
        q_offsets.append(q_cursor)
        page_cursor += pages
        row_cursor += p + q
        q_cursor += q

    return PrefixMergeLayout(
        prefix_lens=tuple(ps),
        seq_lens=tuple(qs),
        pages_per_req=pages_per,
        page_offsets=tuple(page_offsets),
        kv_offsets=tuple(kv_offsets),
        q_offsets=tuple(q_offsets),
        total_rows=row_cursor,
        total_pages=pages_total,
        sum_prefix=sum_p,
        sum_seq=sum_q,
    )


def _shared_meta(*tensors: torch.Tensor):
    first = tensors[0]
    # Pure packing: all four sources must be floating tensors with one dtype
    # and device.  The eligible model route separately requires the pool /
    # split-weight BF16/FP16 dtype before reaching here.
    if not first.dtype.is_floating_point:
        return None
    for tensor in tensors[1:]:
        if tensor.dtype != first.dtype or tensor.device != first.device:
            return None
    return first.dtype, first.device


def pack_merged_rows(
    history_latent: torch.Tensor,
    history_rope: torch.Tensor,
    current_latent: torch.Tensor,
    current_rope: torch.Tensor,
    layout: PrefixMergeLayout,
    *,
    page_size: int = _PAGE,
    latent_dim: int = _KEY_LATENT,
    rope_dim: int = _KEY_ROPE,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Pack history/current latent+rotated-RoPE rows in ``[p, c]`` order.

    Inputs are already flattened per token: history latent/rope hold exactly
    ``total_pages * page_size`` rows (trailing page padding included), current
    latent/rope hold exactly ``sum_seq`` rows.  The returned tensors are fresh
    contiguous ``[total_rows, latent_dim]`` and ``[total_rows, 1, rope_dim]``;
    the sources are never mutated.
    """
    hist_rows = layout.total_pages * page_size
    cur_rows = layout.sum_seq
    if history_latent.numel() != hist_rows * latent_dim:
        return None
    if history_rope.numel() != hist_rows * rope_dim:
        return None
    if current_latent.numel() != cur_rows * latent_dim:
        return None
    if current_rope.numel() != cur_rows * rope_dim:
        return None
    meta = _shared_meta(
        history_latent, history_rope, current_latent, current_rope
    )
    if meta is None:
        return None
    dtype, device = meta

    merged_latent = torch.empty(
        (layout.total_rows, latent_dim), dtype=dtype, device=device
    )
    merged_rope = torch.empty(
        (layout.total_rows, 1, rope_dim), dtype=dtype, device=device
    )

    hist_lat = history_latent.reshape(hist_rows, latent_dim)
    hist_rope = history_rope.reshape(hist_rows, 1, rope_dim)
    cur_lat = current_latent.reshape(cur_rows, latent_dim)
    cur_rope = current_rope.reshape(cur_rows, 1, rope_dim)

    for j, (p, q) in enumerate(zip(layout.prefix_lens, layout.seq_lens)):
        kv_start = layout.kv_offsets[j]
        if p:
            src = layout.page_offsets[j] * page_size
            merged_latent[kv_start : kv_start + p].copy_(hist_lat[src : src + p])
            merged_rope[kv_start : kv_start + p].copy_(hist_rope[src : src + p])
        if q:
            q_start = layout.q_offsets[j]
            row = kv_start + p
            merged_latent[row : row + q].copy_(cur_lat[q_start : q_start + q])
            merged_rope[row : row + q].copy_(cur_rope[q_start : q_start + q])
    return merged_latent, merged_rope


def merge_prefix_and_current_rows(
    history_latent: torch.Tensor,
    history_rope: torch.Tensor,
    current_latent: torch.Tensor,
    current_rope: torch.Tensor,
    prefix_lens: Optional[Sequence[int]],
    seq_lens: Optional[Sequence[int]],
    num_selected_pages: int,
    *,
    page_size: int = _PAGE,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, PrefixMergeLayout]]:
    """Build the layout and pack in one call (shared by the glue and tests)."""
    layout = build_merge_layout(
        prefix_lens, seq_lens, num_selected_pages, page_size=page_size
    )
    if layout is None:
        return None
    packed = pack_merged_rows(
        history_latent,
        history_rope,
        current_latent,
        current_rope,
        layout,
        page_size=page_size,
    )
    if packed is None:
        return None
    return packed[0], packed[1], layout


# ---------------------------------------------------------------------------
# per-call context
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MergedPrefixKVProjection:
    """Projected full K/V plus host intervals, passed prepare -> core -> backend.

    The dataclass is a value for exactly one layer/forward; it is never stored
    on the attention module, backend or ``ForwardBatch``.
    """

    k_full: torch.Tensor
    v_full: torch.Tensor
    prefix_lens: Tuple[int, ...]
    seq_lens: Tuple[int, ...]
    kv_offsets: Tuple[int, ...]
    q_offsets: Tuple[int, ...]
    total_rows: int
    num_heads: int
    qk_head_dim: int
    v_head_dim: int
    layer_id: int


def iter_fia_request_slices(context: MergedPrefixKVProjection):
    """Yield ``(req, q_start, kv_start, kv_end, q_len)`` in request order.

    ``kv_start:kv_end`` covers exactly ``[prefix_j, current_j]``; the query
    rows are always ``q_start:q_start+q_len``.  Zero-query requests are
    yielded with ``q_len == 0`` so callers can skip FIA while the independent
    KV offsets stay correct.
    """
    if not (
        len(context.prefix_lens)
        == len(context.seq_lens)
        == len(context.kv_offsets)
        == len(context.q_offsets)
    ):
        raise ValueError("merged prefix KV host metadata lengths disagree")
    for req_idx, (q_len, prefix_len) in enumerate(
        zip(context.seq_lens, context.prefix_lens)
    ):
        q_start = context.q_offsets[req_idx]
        kv_start = context.kv_offsets[req_idx]
        yield req_idx, q_start, kv_start, kv_start + prefix_len + q_len, q_len


# ---------------------------------------------------------------------------
# glue used by forward_mha_prepare_npu
# ---------------------------------------------------------------------------
def _attn_geometry_matches(attn: object, layer: object) -> bool:
    if getattr(attn, "use_dsa", False):
        return False
    if not getattr(attn, "use_deepseek_yarn_rope", False):
        return False
    if getattr(attn, "kv_lora_rank", None) != _KEY_LATENT:
        return False
    if getattr(attn, "qk_nope_head_dim", None) != _KEY_NOPE:
        return False
    if getattr(attn, "qk_rope_head_dim", None) != _KEY_ROPE:
        return False
    if getattr(attn, "v_head_dim", None) != _VALUE_DIM:
        return False
    heads = getattr(attn, "num_local_heads", None)
    if not isinstance(heads, int) or heads <= 0:
        return False
    if getattr(layer, "qk_head_dim", None) != _QK_HEAD_DIM:
        return False
    if getattr(layer, "v_head_dim", None) != _VALUE_DIM:
        return False
    if getattr(layer, "tp_q_head_num", None) != heads:
        return False
    # The historical path views the packed kv_b_proj output with the KV head
    # count, so a differing KV head count cannot be merged into this geometry.
    if getattr(layer, "tp_k_head_num", None) != heads:
        return False
    if getattr(layer, "tp_v_head_num", None) != heads:
        return False
    if getattr(layer, "is_cross_attention", False):
        return False
    if getattr(layer, "sliding_window_size", -1) != -1:
        return False
    if getattr(layer, "layer_id", None) != getattr(attn, "layer_id", None):
        return False
    return True


def _pool_layout_matches(pool: object) -> bool:
    if not bool(getattr(pool, "unified_kv", False)):
        return False
    if bool(getattr(pool, "use_dsa", False)):
        return False
    if getattr(pool, "kv_lora_rank", None) != _KEY_LATENT:
        return False
    if getattr(pool, "qk_rope_head_dim", None) != _KEY_ROPE:
        return False
    if getattr(pool, "page_size", None) != _PAGE:
        return False
    if getattr(pool, "prefill_expanded_geometry", None) is not None:
        return False
    dtype = getattr(pool, "dtype", None)
    if dtype not in _SUPPORTED_DTYPES:
        return False
    if getattr(pool, "store_dtype", None) != dtype:
        return False
    return True


def _forward_batch_eligible(forward_batch: object, backend: object) -> bool:
    try:
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        if forward_batch.forward_mode != ForwardMode.EXTEND:
            return False
    except Exception:
        return False
    if getattr(forward_batch, "attn_cp_metadata", None) is not None:
        return False
    if getattr(forward_batch, "encoder_lens", None) is not None:
        return False
    if getattr(backend, "attn_cp_size", 1) != 1:
        return False
    if getattr(backend, "graph_mode", False):
        return False
    if getattr(backend, "enable_torch_compile", False):
        return False
    if getattr(backend, "is_dllm_model", False):
        return False
    try:
        from sglang.srt.hardware_backend.npu.attention.mla_preprocess import (
            is_mla_preprocess_for_batch,
        )

        if is_mla_preprocess_for_batch(forward_batch):
            return False
    except ImportError:
        return False
    return True


def _current_tensors_ok(
    kv_a: object,
    k_pe: object,
    q: object,
    *,
    sum_seq: int,
    num_heads: int,
    dtype: torch.dtype,
    device: torch.device,
) -> bool:
    if not torch.is_tensor(kv_a) or not torch.is_tensor(k_pe) or not torch.is_tensor(q):
        return False
    if kv_a.dim() not in (3, 4) or kv_a.shape[-1] != _KEY_LATENT:
        return False
    if kv_a.dtype != dtype or kv_a.device != device or not kv_a.is_contiguous():
        return False
    if kv_a.numel() != kv_a.shape[0] * _KEY_LATENT:
        return False
    if k_pe.dim() < 2 or k_pe.shape[-1] != _KEY_ROPE:
        return False
    if k_pe.dtype != dtype or k_pe.device != device or not k_pe.is_contiguous():
        return False
    if k_pe.shape[0] != kv_a.shape[0] or k_pe.numel() != k_pe.shape[0] * _KEY_ROPE:
        return False
    if q.dim() != 3 or q.shape[1] != num_heads or q.shape[2] != _QK_HEAD_DIM:
        return False
    if q.dtype != dtype or q.device != device:
        return False
    if q.shape[0] != kv_a.shape[0]:
        return False
    # The dedicated consumer allocates the attention output with q.shape[0]
    # rows but only fills rows covered by host query lengths.  A padded batch
    # would therefore expose uninitialised output rows; this focused eager
    # route declines it and leaves the original path responsible.
    if q.shape[0] != sum_seq:
        return False
    return True


def maybe_merge_prefix_kv_b_proj(
    attn: object,
    module: object,
    kv_a: torch.Tensor,
    k_pe: torch.Tensor,
    q: torch.Tensor,
    forward_batch: object,
    *,
    unified_cache_ready: bool,
    backend: object = None,
    pool: object = None,
) -> Optional[MergedPrefixKVProjection]:
    """Try to merge prefix/current rows and project them with two GEMMs.

    Returns ``None`` before any contract change when unsupported.  ``backend``
    and ``pool`` are injectable for CPU tests; production reads them from the
    existing forward context.
    """
    if not merge_prefix_kv_b_proj_enabled():
        return None
    if not unified_cache_ready:
        return None
    if forward_batch is None:
        return None
    if _graph_or_compile_active():
        return None

    layer = getattr(attn, "attn_mha", None)
    if layer is None:
        return None
    if getattr(layer, "kv_b_proj", None) is not module:
        # The backend's historical projection must be the same module whose
        # split state we reuse.
        return None
    if not _attn_geometry_matches(attn, layer):
        return None
    concat_k = getattr(attn, "_concat_and_cast_mha_k", None)
    if concat_k is None:
        return None
    num_heads = attn.num_local_heads

    if backend is None:
        from sglang.srt.model_executor.forward_context import get_attn_backend

        backend = get_attn_backend()
    if not _backend_is_supported(backend):
        return None
    if pool is None:
        from sglang.srt.model_executor.forward_context import get_token_to_kv_pool

        pool = get_token_to_kv_pool()
    if getattr(backend, "token_to_kv_pool", None) is not pool:
        return None
    if not getattr(backend, "use_mla", False):
        return None
    if not _pool_layout_matches(pool):
        return None
    if not _forward_batch_eligible(forward_batch, backend):
        return None

    metadata = getattr(backend, "forward_metadata", None)
    if metadata is None:
        return None
    flatten_pages = getattr(metadata, "flatten_prefix_block_tables", None)
    if flatten_pages is None or not torch.is_tensor(flatten_pages):
        return None
    metadata_prefix_lens = getattr(metadata, "prefix_lens", None)
    metadata_seq_lens = getattr(metadata, "extend_seq_lens_cpu_int", None)
    if not torch.is_tensor(metadata_prefix_lens):
        return None
    if not torch.is_tensor(metadata_seq_lens):
        return None

    # Reuse the split module's host-metadata readiness (weight currentness,
    # quant method, hooks, SP, dtype/device guards) WITHOUT running a dummy
    # current GEMM.
    from sglang.srt.hardware_backend.npu.modules.split_kv_b_proj import (
        maybe_forward_split_kv_b_proj,
        split_kv_b_proj_state_for_forward,
    )

    state = split_kv_b_proj_state_for_forward(attn, module, forward_batch)
    if state is None:
        return None
    if state.tp_size != 1:
        return None
    if state.num_heads != num_heads:
        return None

    layout = build_merge_layout(
        forward_batch.extend_prefix_lens_cpu,
        forward_batch.extend_seq_lens_cpu,
        flatten_pages.numel(),
    )
    if layout is None:
        return None
    if layout.total_rows == 0:
        return None
    if len(layout.prefix_lens) != int(metadata_prefix_lens.numel()):
        return None
    if len(layout.seq_lens) != int(metadata_seq_lens.numel()):
        return None
    if not _current_tensors_ok(
        kv_a,
        k_pe,
        q,
        sum_seq=layout.sum_seq,
        num_heads=num_heads,
        dtype=state.weight_dtype,
        device=state.weight_device,
    ):
        return None

    # Page-first gather: preserves page order/duplicates/layer offsets and the
    # public getter's HiCache/layer-transfer wait; returns None when any of its
    # own gates is unsupported, in which case the original path owns the work.
    from sglang.srt.hardware_backend.npu.modules.prefix_gather_unified_cache_npu import (
        maybe_gather_unified_prefix_kv,
    )

    gathered = maybe_gather_unified_prefix_kv(
        pool, int(attn.layer_id), flatten_pages
    )
    if gathered is None:
        return None
    history_latent, history_rope = gathered
    if not history_latent.is_contiguous() or not history_rope.is_contiguous():
        return None
    expected_hist_rows = layout.total_pages * _PAGE
    if history_latent.numel() != expected_hist_rows * _KEY_LATENT:
        return None
    if history_rope.numel() != expected_hist_rows * _KEY_ROPE:
        return None

    current_latent = kv_a.reshape(-1, _KEY_LATENT)[: layout.sum_seq]
    current_rope = k_pe.reshape(-1, _KEY_ROPE)[: layout.sum_seq]
    packed = pack_merged_rows(
        history_latent,
        history_rope,
        current_latent,
        current_rope,
        layout,
    )
    if packed is None:
        return None
    merged_latent, merged_rope = packed
    if not merged_latent.is_contiguous() or not merged_rope.is_contiguous():
        return None

    # Exactly two KV-B GEMMs for all prefix + current rows, reusing the
    # existing split helper (weight-currentness, hooks, SP/dtype guards, bias).
    split_kv = maybe_forward_split_kv_b_proj(attn, module, merged_latent, forward_batch)
    if split_kv is None:
        # Cannot use the prepared split state: take the old path rather than
        # claim a merge that did not happen (no extra projection was run).
        return None
    k_nope, v_full = split_kv
    if (
        k_nope.shape != (layout.total_rows, num_heads, _KEY_NOPE)
        or v_full.shape != (layout.total_rows, num_heads, _VALUE_DIM)
    ):
        raise RuntimeError(
            "merged split kv_b_proj returned unexpected shapes "
            f"K={tuple(k_nope.shape)} V={tuple(v_full.shape)} "
            f"expected ({layout.total_rows}, {num_heads}, {_KEY_NOPE}/"
            f"{_VALUE_DIM})"
        )
    if not k_nope.is_contiguous() or not v_full.is_contiguous():
        raise RuntimeError("merged split kv_b_proj output is not contiguous")

    k_full = concat_k(k_nope, merged_rope, forward_batch)
    if k_full.shape != (layout.total_rows, num_heads, _QK_HEAD_DIM):
        raise RuntimeError(
            f"merged K assembly returned {tuple(k_full.shape)}, expected "
            f"({layout.total_rows}, {num_heads}, {_QK_HEAD_DIM})"
        )
    if not k_full.is_contiguous():
        raise RuntimeError("merged K assembly output is not contiguous")

    return MergedPrefixKVProjection(
        k_full=k_full,
        v_full=v_full,
        prefix_lens=layout.prefix_lens,
        seq_lens=layout.seq_lens,
        kv_offsets=layout.kv_offsets,
        q_offsets=layout.q_offsets,
        total_rows=layout.total_rows,
        num_heads=num_heads,
        qk_head_dim=_QK_HEAD_DIM,
        v_head_dim=_VALUE_DIM,
        layer_id=int(attn.layer_id),
    )
