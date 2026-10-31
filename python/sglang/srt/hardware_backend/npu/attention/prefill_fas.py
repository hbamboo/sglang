"""Opt-in eager FAS for a measured subset of expanded PD-prefill batches."""

from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PrefillFASDecision:
    owner: object
    reason: str
    q_lengths: tuple = ()
    kv_lengths: tuple = ()
    q_cumulative: tuple = ()
    kv_cumulative: tuple = ()

    @property
    def eligible(self):
        return self.reason == "eligible"


@dataclass
class PrefillFASStaging:
    owner: object
    block_tables: object
    q_lengths: tuple
    kv_lengths: tuple
    slots: object
    producer_stream: object
    ready_event: object = None


def _cumulative(lengths):
    total = 0
    result = []
    for length in lengths:
        total += length
        result.append(total)
    return tuple(result)


def _shape_reason(q_lengths, prefixes, tokens):
    """Conservative initial subset; it is not the full business workload."""
    if len(q_lengths) != len(prefixes) or not q_lengths:
        return "length_metadata"
    if any(q <= 0 or p < 0 for q, p in zip(q_lengths, prefixes)):
        return "length_metadata"
    if sum(q_lengths) != tokens:
        return "length_metadata"
    if not any(prefixes):
        return "fresh_unchanged"
    if not 2 <= len(q_lengths) <= 8:
        return "batch_size"
    if not 8192 <= tokens <= 32768:
        return "total_tokens"
    if any(not 512 <= q <= 6528 for q in q_lengths):
        return "extend_length"
    if any(p > 640 for p in prefixes):
        return "prefix_length"
    # An aggregate guard permits mixed short tails while limiting the
    # staging/dispatch cost per new token. Per-request prefix ratio and
    # squared-query density did not separate the measured profit cases.
    if tokens < 2048 * len(q_lengths):
        return "mean_query_length"
    return "eligible"


def _get_step_decision(backend, metadata, owner, build):
    """The eager backend constructs a new ForwardMetadata for every forward."""
    decision = metadata.prefill_fas_decision
    if decision is not None and decision.owner is owner:
        return decision
    decision = build()
    metadata.prefill_fas_decision = decision
    counts = getattr(backend, "prefill_fas_step_reason_counts", None)
    if counts is None:
        counts = {}
        backend.prefill_fas_step_reason_counts = counts
    counts[decision.reason] = counts.get(decision.reason, 0) + 1
    if decision.eligible and not getattr(backend, "_prefill_fas_first_hit_logged", False):
        backend._prefill_fas_first_hit_logged = True
        logger.info(
            "SGLANG_NPU_PREFILL_FAS first eligible eager step: batch=%s, "
            "new_tokens=%s; only the guarded subset uses FAS",
            len(decision.q_lengths),
            sum(decision.q_lengths),
        )
    return decision


def prefill_fas_step_counts(backend):
    """Read-only snapshot for validation/profiling; counts eager attempts, not layers.

    Compiled/captured forwards do not execute this Python dispatch on every replay
    and are excluded from this denominator. Use scheduler trace counts as the
    denominator when reporting coverage of the entire workload.
    """
    return dict(getattr(backend, "prefill_fas_step_reason_counts", {}))


def _get_step_staging(metadata, decision, block_tables, build):
    cached = metadata.prefill_fas_staging
    if (
        cached is not None
        and cached.owner is decision.owner
        and cached.block_tables is block_tables
        and cached.q_lengths == decision.q_lengths
        and cached.kv_lengths == decision.kv_lengths
    ):
        return cached
    # No backend/global slots cache: even identical lengths rebuild next step.
    cached = build()
    metadata.prefill_fas_staging = cached
    return cached


def _slots_for_stream(staging, stream):
    if stream == staging.producer_stream:
        return
    # The readiness event is recorded at production time, before subsequent
    # producer work can introduce reverse dependencies on a consumer stream.
    # Ordinary same-stream calls wait on no event and never synchronize.
    if staging.ready_event is None:
        raise RuntimeError("FAS step slots have no producer readiness event")
    stream.wait_event(staging.ready_event)
    staging.slots.record_stream(stream)


def _build_staging(decision, block_tables, page_size):
    import torch

    width = block_tables.shape[1]
    table_indices_cpu = []
    in_page_cpu = []
    for batch, length in enumerate(decision.kv_lengths):
        positions = torch.arange(length, dtype=torch.int64)
        table_indices_cpu.append(positions // page_size + batch * width)
        in_page_cpu.append(positions % page_size)
    # O(sumKV) metadata setup is done once per step, shared by all layers.
    # Every layer still gathers its own K/V payload into fresh dense buffers.
    indices = torch.cat(table_indices_cpu).to(block_tables.device)
    offsets = torch.cat(in_page_cpu).to(block_tables.device)
    physical_blocks = block_tables.reshape(-1).index_select(0, indices).to(torch.int64)
    slots = physical_blocks * page_size + offsets
    producer_stream = torch.npu.current_stream()
    ready_event = torch.npu.Event()
    ready_event.record(producer_stream)
    return PrefillFASStaging(
        owner=decision.owner,
        block_tables=block_tables,
        q_lengths=decision.q_lengths,
        kv_lengths=decision.kv_lengths,
        slots=slots,
        producer_stream=producer_stream,
        ready_event=ready_event,
    )


def maybe_forward_expanded_prefill_fas(backend, query, layer, forward_batch):
    """Return None for ordinary FIA fallback; device/API errors propagate."""
    import torch
    import torch_npu

    from sglang.srt.runtime_context import get_disagg
    from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
        get_tc_piecewise_forward_context,
    )

    # Never mutate Python metadata or run FAS inside compiled/captured graphs.
    # This also excludes warmups/replays that use persistent graph metadata.
    if (
        torch.compiler.is_compiling()
        or get_tc_piecewise_forward_context() is not None
        or backend.graph_mode
        or torch.npu.is_current_stream_capturing()
    ):
        return None

    metadata = backend.forward_metadata
    pool = backend.token_to_kv_pool

    def decide():
        def fallback(reason):
            return PrefillFASDecision(owner=forward_batch, reason=reason)

        if get_disagg().disaggregation_mode != "prefill":
            return fallback("not_pd_prefill")
        if (
            not forward_batch.forward_mode.is_extend()
            or forward_batch.forward_mode.is_target_verify()
            or forward_batch.forward_mode.is_draft_extend_v2()
        ):
            return fallback("forward_mode")
        if query.dtype != torch.bfloat16 or query.device.type != "npu" or not query.is_contiguous():
            return fallback("query_dtype_layout")
        if (
            layer.tp_q_head_num != 20
            or layer.tp_k_head_num != 20
            or layer.qk_head_dim != 256
            or layer.v_head_dim != 256
            or pool.prefill_expanded_geometry is None
            or pool.page_size != 128
        ):
            return fallback("geometry")
        q_lengths = tuple(int(v) for v in forward_batch.extend_seq_lens_cpu)
        prefixes = tuple(int(v) for v in forward_batch.extend_prefix_lens_cpu)
        reason = _shape_reason(q_lengths, prefixes, forward_batch.num_token_non_padded_cpu)
        if reason != "eligible":
            return fallback(reason)
        kv_lengths = tuple(int(v) for v in backend._get_seq_lens_cpu_int_list())
        q_cumulative = _cumulative(q_lengths)
        if (
            kv_lengths != tuple(q + p for q, p in zip(q_lengths, prefixes))
            or tuple(metadata.seq_lens_list_cumsum) != q_cumulative
            or tuple(query.shape) != (sum(q_lengths), 20, 256)
        ):
            return fallback("length_metadata")
        block_tables = metadata.block_tables
        if (
            block_tables is None
            or block_tables.ndim != 2
            or block_tables.shape[0] != len(q_lengths)
            or block_tables.shape[1] < (max(kv_lengths) + 127) // 128
            or block_tables.device != query.device
        ):
            return fallback("block_table")
        key, value = pool.get_prefill_expanded_kv(layer.layer_id)
        expected_tail = (128, 20, 256)
        if any(
            cache.dtype != torch.bfloat16
            or cache.device != query.device
            or cache.ndim != 4
            or tuple(cache.shape[1:]) != expected_tail
            or not cache.is_contiguous()
            for cache in (key, value)
        ):
            return fallback("cache_dtype_layout")
        return PrefillFASDecision(
            owner=forward_batch,
            reason="eligible",
            q_lengths=q_lengths,
            kv_lengths=kv_lengths,
            q_cumulative=q_cumulative,
            kv_cumulative=_cumulative(kv_lengths),
        )

    decision = _get_step_decision(backend, metadata, forward_batch, decide)
    if not decision.eligible:
        return None
    # Layer-local dtype/layout must remain guarded even after the step decision
    # was cached; a mixed layer cannot inherit the first layer's eligibility.
    if (
        query.dtype != torch.bfloat16
        or query.device.type != "npu"
        or not query.is_contiguous()
        or tuple(query.shape) != (sum(decision.q_lengths), 20, 256)
        or (layer.tp_q_head_num, layer.tp_k_head_num, layer.qk_head_dim, layer.v_head_dim)
        != (20, 20, 256, 256)
    ):
        return None
    key, value = pool.get_prefill_expanded_kv(layer.layer_id)
    if any(
        cache.dtype != torch.bfloat16
        or cache.device != query.device
        or cache.ndim != 4
        or tuple(cache.shape[1:]) != (128, 20, 256)
        or not cache.is_contiguous()
        for cache in (key, value)
    ):
        return None
    staging = _get_step_staging(
        metadata,
        decision,
        metadata.block_tables,
        lambda: _build_staging(decision, metadata.block_tables, pool.page_size),
    )
    _slots_for_stream(staging, torch.npu.current_stream())
    dense_key = key.reshape(-1, 20, 256).index_select(0, staging.slots)
    dense_value = value.reshape(-1, 20, 256).index_select(0, staging.slots)
    # Public FAS always constructs all seven outputs, including FP32 max/sum.
    # There is no private kernel call or suppression of mandatory outputs.
    outputs = torch_npu.npu_fusion_attention(
        query,
        dense_key,
        dense_value,
        head_num=20,
        input_layout="TND",
        atten_mask=backend.fia_mask,
        scale=layer.scaling,
        keep_prob=1.0,
        actual_seq_qlen=decision.q_cumulative,
        actual_seq_kvlen=decision.kv_cumulative,
        sparse_mode=3,
        pre_tockens=2147483647,
        next_tockens=0,
    )
    return outputs[0]
