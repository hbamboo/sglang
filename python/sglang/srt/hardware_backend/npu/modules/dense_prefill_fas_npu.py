"""Opt-in dense FAS for eager GLM prefill on the latent (non-expanded) KV path.

``SGLANG_NPU_PREFILL_FAS=1`` historically drives the expanded-KV prefill route,
which stages K/V through ``get_prefill_expanded_kv`` and two ``index_select``
gathers.  This module lets the same public ``torch_npu.npu_fusion_attention``
(TND, sparse mode 3) run directly on the already-dense tensors of the latent
route, with ``expandKV=0``:

* cold/no-prefix ordinary MHA core: the current ``q/k/v`` materialised by the
  existing prepare and cache writer;
* merged prefix/current KV-B route: ``context.k_full`` / ``context.v_full`` are
  already the per-request ``[prefix, current]`` rows in FIA order.

No expanded-cache allocation, gather, index_select, concat or full-pool read is
introduced.  Inputs must already be contiguous; the module never calls
``.contiguous()`` or copies to qualify.  Unsupported cases return ``None`` and
the existing FIA / RadixAttention path is preserved exactly.

Eligibility is deliberately narrow and checked from real tensors per layer;
only the host cumulative-length plan is cached, once per scheduler step in the
fresh ``ForwardMetadata`` (keyed by the actual host length lists).  There is no
cross-step cache and no per-layer device scalar read.
"""

from __future__ import annotations

import logging
import numbers
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch

logger = logging.getLogger(__name__)

_DENSE_FAS_CALLS = 0
_HEADS = 20
_HEAD_DIM = 256


def dense_fas_call_count() -> int:
    """Actual public-FAS submissions (not eligibility decisions)."""
    return _DENSE_FAS_CALLS


def reset_dense_fas_call_count() -> None:
    """Test/profiling hook."""
    global _DENSE_FAS_CALLS
    _DENSE_FAS_CALLS = 0


@dataclass(frozen=True)
class DenseFASPlan:
    q_lengths: Tuple[int, ...]
    prefix_lengths: Tuple[int, ...]
    q_cumulative: Tuple[int, ...]
    kv_cumulative: Tuple[int, ...]
    total_q: int
    total_kv: int


def _host_ints(values: Optional[Sequence[int]]) -> Optional[Tuple[int, ...]]:
    if values is None:
        return None
    result = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, numbers.Integral):
            return None
        result.append(int(value))
    return tuple(result)


def _cumulative(lengths: Sequence[int]) -> Tuple[int, ...]:
    total = 0
    result = []
    for length in lengths:
        total += length
        result.append(total)
    return tuple(result)


def build_dense_fas_plan(
    prefix_values: Optional[Sequence[int]],
    seq_values: Optional[Sequence[int]],
    *,
    q_rows: int,
    kv_rows: int,
) -> Optional[DenseFASPlan]:
    """Validate host lengths/layout and build the ragged cumulative plan."""
    prefixes = _host_ints(prefix_values)
    seqs = _host_ints(seq_values)
    if prefixes is None or seqs is None or not seqs or len(prefixes) != len(seqs):
        return None
    if any(q <= 0 for q in seqs) or any(p < 0 for p in prefixes):
        return None
    if not isinstance(q_rows, int) or not isinstance(kv_rows, int):
        return None
    total_q = sum(seqs)
    kv_lengths = tuple(p + q for p, q in zip(prefixes, seqs))
    total_kv = sum(kv_lengths)
    if total_q != q_rows or total_kv != kv_rows:
        return None
    return DenseFASPlan(
        q_lengths=seqs,
        prefix_lengths=prefixes,
        q_cumulative=_cumulative(seqs),
        kv_cumulative=_cumulative(kv_lengths),
        total_q=total_q,
        total_kv=total_kv,
    )


def _step_plan(
    backend: object,
    prefixes: Optional[Sequence[int]],
    seqs: Optional[Sequence[int]],
    *,
    q_rows: int,
    kv_rows: int,
) -> Optional[DenseFASPlan]:
    """Per-step cache keyed by the actual host lengths; metadata is fresh/step."""
    metadata = getattr(backend, "forward_metadata", None)
    if metadata is None:
        return None
    key = (tuple(prefixes) if prefixes is not None else None,
           tuple(seqs) if seqs is not None else None, q_rows, kv_rows)
    if getattr(metadata, "dense_fas_plan_key", None) == key:
        return getattr(metadata, "dense_fas_plan", None)
    plan = build_dense_fas_plan(prefixes, seqs, q_rows=q_rows, kv_rows=kv_rows)
    metadata.dense_fas_plan = plan
    metadata.dense_fas_plan_key = key
    return plan


def _fas_enabled() -> bool:
    from sglang.srt.environ import envs

    return envs.SGLANG_NPU_PREFILL_FAS.get()


def _graph_or_compile_active(backend: object) -> bool:
    try:
        if torch.compiler.is_compiling():
            return True
    except Exception:
        return True
    try:
        from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
            get_tc_piecewise_forward_context,
        )

        if get_tc_piecewise_forward_context() is not None:
            return True
    except Exception:
        return True
    if getattr(backend, "graph_mode", False):
        return True
    npu = getattr(torch, "npu", None)
    capture = getattr(npu, "is_current_stream_capturing", None) if npu else None
    if capture is not None:
        try:
            if capture():
                return True
        except Exception:
            return True
    return False


def _backend_is_supported(backend: object) -> bool:
    try:
        from sglang.srt.hardware_backend.npu.attention.ascend_backend import (
            AscendAttnBackend,
        )
    except Exception:
        return False
    return isinstance(backend, AscendAttnBackend)


def _pa_nz_active() -> bool:
    from sglang.srt.hardware_backend.npu.attention.mla_preprocess import is_fia_nz

    return bool(is_fia_nz())


def _scatter_active() -> bool:
    try:
        from sglang.srt.layers.communicator import get_attn_tp_context

        return bool(get_attn_tp_context().input_scattered)
    except Exception:
        return True


def _invoke_public_fas(q, k, v, **kwargs):
    """Single dispatch point so tests can observe the public-FAS call."""
    import torch_npu

    return torch_npu.npu_fusion_attention(q, k, v, **kwargs)


def _common_reason(
    backend: object,
    layer: object,
    forward_batch: object,
    q: object,
    k: object,
    v: object,
    expected_kv_rows: int,
) -> Optional[str]:
    if not _fas_enabled():
        return "disabled"
    if _graph_or_compile_active(backend):
        return "graph_or_compile"
    if not _backend_is_supported(backend):
        return "backend"
    try:
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        if getattr(forward_batch, "forward_mode", None) != ForwardMode.EXTEND:
            return "forward_mode"
    except Exception:
        return "forward_mode"
    if getattr(forward_batch, "spec_info", None) is not None:
        return "speculative"
    if getattr(forward_batch, "attn_cp_metadata", None) is not None:
        return "cp"
    if getattr(backend, "attn_cp_size", 1) != 1:
        return "cp"
    if _scatter_active():
        return "dp_scatter"
    if _pa_nz_active():
        return "pa_nz"
    pool = getattr(backend, "token_to_kv_pool", None)
    if pool is None or not bool(getattr(pool, "unified_kv", False)):
        return "pool"
    if bool(getattr(pool, "use_dsa", False)):
        return "dsa"
    if getattr(pool, "prefill_expanded_geometry", None) is not None:
        return "expanded_kv"
    if getattr(layer, "is_cross_attention", False):
        return "cross_attention"
    if getattr(layer, "sliding_window_size", -1) != -1:
        return "sliding_window"
    if (
        getattr(layer, "tp_q_head_num", None),
        getattr(layer, "tp_k_head_num", None),
        getattr(layer, "tp_v_head_num", None),
        getattr(layer, "qk_head_dim", None),
        getattr(layer, "v_head_dim", None),
    ) != (_HEADS, _HEADS, _HEADS, _HEAD_DIM, _HEAD_DIM):
        return "geometry"
    if not (torch.is_tensor(q) and torch.is_tensor(k) and torch.is_tensor(v)):
        return "tensor"
    if q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype:
        return "dtype"
    if q.device != k.device or q.device != v.device:
        return "device"
    if not (q.is_contiguous() and k.is_contiguous() and v.is_contiguous()):
        return "layout"
    expected = (expected_kv_rows, _HEADS, _HEAD_DIM)
    if q.dim() != 3 or tuple(q.shape[1:]) != (_HEADS, _HEAD_DIM):
        return "q_shape"
    if k.shape != expected or v.shape != expected:
        return "kv_shape"
    mask = getattr(backend, "fia_mask", None)
    if mask is None or getattr(mask, "device", None) != q.device:
        return "mask"
    q_rows = q.shape[0]
    padded = getattr(forward_batch, "num_token_non_padded_cpu", q_rows)
    if padded is not None and padded != q_rows:
        return "query_padding"
    return None


def _run(backend, q, k, v, layer, plan: DenseFASPlan):
    global _DENSE_FAS_CALLS
    with torch.profiler.record_function("npu_prefill_dense_fas"):
        outputs = _invoke_public_fas(
            q,
            k,
            v,
            head_num=_HEADS,
            input_layout="TND",
            atten_mask=backend.fia_mask,
            scale=layer.scaling,
            keep_prob=1.0,
            actual_seq_qlen=plan.q_cumulative,
            actual_seq_kvlen=plan.kv_cumulative,
            sparse_mode=3,
            pre_tockens=2147483647,
            next_tockens=0,
        )
    # Count the actual native submission before validating its outputs.
    _DENSE_FAS_CALLS += 1
    if _DENSE_FAS_CALLS == 1:
        logger.info(
            "SGLANG_NPU_PREFILL_FAS first dense execution: layer=%s batch=%s "
            "new_tokens=%s prefixes=%s",
            getattr(layer, "layer_id", None),
            len(plan.q_lengths),
            plan.total_q,
            plan.prefix_lengths,
        )
    # The public API always constructs all seven outputs (attention result plus
    # FP32 max/sum and lse); no private kernel and no output suppression.
    if not isinstance(outputs, (tuple, list)) or len(outputs) != 7:
        raise RuntimeError(
            "public npu_fusion_attention must return all seven outputs, got "
            f"{type(outputs).__name__} of length "
            f"{len(outputs) if hasattr(outputs, '__len__') else 'n/a'}"
        )
    attn = outputs[0]
    if (
        not torch.is_tensor(attn)
        or not attn.is_contiguous()
        or attn.dim() != 3
        or tuple(attn.shape) != (plan.total_q, _HEADS, _HEAD_DIM)
    ):
        raise RuntimeError(
            "dense FAS returned an unexpected attention result "
            f"(shape={getattr(attn, 'shape', None)}, contiguous="
            f"{getattr(attn, 'is_contiguous', lambda: None)()})"
        )
    # TND output is contiguous; this is a view, never a copy.
    return attn.view(-1, _HEADS * _HEAD_DIM)


def maybe_forward_dense_prefill_fas(
    backend: object,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layer: object,
    forward_batch: object,
) -> Optional[torch.Tensor]:
    """Cold/no-prefix ordinary MHA core on already-dense q/k/v.

    Returns a ``[new_tokens, 20*256]`` view or ``None`` (existing RadixAttention
    path).  Batches with any historical prefix are intentionally rejected here;
    the eligible merged context route covers prefix cases without dropping
    history.
    """
    prefixes = getattr(forward_batch, "extend_prefix_lens_cpu", None)
    seqs = getattr(forward_batch, "extend_seq_lens_cpu", None)
    host_prefixes = _host_ints(prefixes)
    if host_prefixes is None or any(host_prefixes):
        return None
    if not torch.is_tensor(q) or not torch.is_tensor(k) or not torch.is_tensor(v):
        return None
    reason = _common_reason(backend, layer, forward_batch, q, k, v, k.shape[0])
    if reason is not None:
        return None
    plan = _step_plan(backend, host_prefixes, seqs, q_rows=q.shape[0], kv_rows=k.shape[0])
    if plan is None:
        return None
    return _run(backend, q, k, v, layer, plan)


def maybe_forward_merged_dense_prefill_fas(
    backend: object,
    q: torch.Tensor,
    context: object,
    layer: object,
    forward_batch: object,
) -> Optional[torch.Tensor]:
    """Merged prefix/current KV-B context on already-dense full K/V.

    ``context.k_full`` / ``context.v_full`` already hold
    ``[prefix_j, current_j]`` per request in FIA order; only view/tensor checks
    and one whole-batch native call are performed.
    """
    try:
        prefixes = tuple(context.prefix_lens)
        seqs = tuple(context.seq_lens)
        q_offsets = tuple(context.q_offsets)
        kv_offsets = tuple(context.kv_offsets)
        kv_rows = int(context.total_rows)
        k_full = context.k_full
        v_full = context.v_full
    except (AttributeError, TypeError, ValueError):
        return None
    if not torch.is_tensor(q) or not torch.is_tensor(k_full) or not torch.is_tensor(v_full):
        return None
    built = build_dense_fas_plan(
        prefixes, seqs, q_rows=q.shape[0], kv_rows=kv_rows
    )
    if built is None:
        return None
    # Context offsets are interval STARTS; both must be the contiguous layout
    # implied by the host lengths (no gaps, no overlap, request order).
    q_starts = (0,) + built.q_cumulative[:-1]
    kv_starts = (0,) + built.kv_cumulative[:-1]
    if q_starts != q_offsets or kv_starts != kv_offsets:
        return None
    reason = _common_reason(backend, layer, forward_batch, q, k_full, v_full, kv_rows)
    if reason is not None:
        return None
    plan = _step_plan(backend, prefixes, seqs, q_rows=q.shape[0], kv_rows=kv_rows)
    if plan is None:
        return None
    return _run(backend, q, k_full, v_full, layer, plan)
