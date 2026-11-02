"""Optional ready-result retirement for ordinary single-rank PD prefill.

No device work, stream switching or transfer-completion polling is added here.
The caller runs on the scheduler stream, outside the next model forward.
"""


def supports_early_prefill_result(scheduler, graph_config, chunked_prefill_size):
    """Static admission, evaluated only when the NPU opt-in flag is enabled."""
    ps = scheduler.ps
    if any(
        getattr(ps, name) != 1
        for name in (
            "tp_size", "pp_size", "dp_size", "attn_tp_size", "attn_cp_size",
            "attn_dcp_size", "attn_dp_size", "moe_ep_size", "moe_dp_size",
        )
    ):
        return False
    if not scheduler.enable_overlap or not scheduler.spec_algorithm.is_none():
        return False
    if chunked_prefill_size != -1:
        return False
    if graph_config is None or graph_config.prefill.backend != "disabled":
        return False
    if any(
        getattr(scheduler, name)
        for name in (
            "enable_pdmux", "enable_staging", "enable_unified_memory",
            "enable_hierarchical_cache", "enable_session_radix_cache",
            "enable_hicache_storage", "enable_hisparse", "enable_dp_attention",
        )
    ):
        return False
    return _supports_cache(scheduler.tree_cache)


def _supports_cache(cache):
    """Only the built-in FULL-only Python tree with a standard paged allocator.

    Imports happen at opt-in admission, after scheduler/cache construction; the
    helper's module import must not trigger the cache registry initialization.
    """
    if (
        type(cache).__module__ != "sglang.srt.mem_cache.unified_radix_cache"
        or type(cache).__name__ != "UnifiedRadixCache"
    ):
        return False
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
    from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeCore
    from sglang.srt.mem_cache.unified_cache.components import ComponentType
    from sglang.srt.mem_cache.unified_cache.components.full_component import FullComponent
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
    from sglang.srt.hardware_backend.npu.allocator_npu import NPUPagedTokenToKVPoolAllocator

    allocator = cache.token_to_kv_pool_allocator
    return (
        type(cache) is UnifiedRadixCache
        and not cache.disable
        and type(cache.tree_core) is UnifiedTreeCore
        and cache._tree_core_backend == "python"
        and not cache.tree_core.is_eagle
        and not cache.tree_core.enable_storage
        and not cache.tree_core.enable_external_cache_linker
        and cache.tree_components == (ComponentType.FULL,)
        and set(cache.components) == {ComponentType.FULL}
        and type(cache.components[ComponentType.FULL]) is FullComponent
        and not cache.enable_session_radix_cache
        and cache.cache_controller is None
        and cache.host_pool_group is None
        and cache._storage_attachment is None
        and cache.linker is None
        and cache.buffer_pipeline is None
        and type(cache.req_to_token_pool) is ReqToTokenPool
        and type(allocator) in (PagedTokenToKVPoolAllocator, NPUPagedTokenToKVPoolAllocator)
        and allocator.need_sort
        and allocator.page_size == cache.page_size == 128
    )


def _ordinary_final_requests(batch):
    if (
        batch.forward_mode.name != "EXTEND"
        or batch.is_dllm()
        or batch.chunked_req is not None
        or batch.spec_info is not None
        or batch.return_logprob
        or not batch.reqs
    ):
        return False
    return all(
        req.inflight_middle_chunks == 0
        and not req.pending_bootstrap
        and req.finished_reason is None
        and req.to_finish is None
        and req.grammar is None
        and req.session is None
        and req.multimodal_inputs is None
        and req.extend_range is not None
        and req.extend_range.end == len(req.origin_input_ids)
        and req.metadata_buffer_index >= 0
        and req.kv.req_pool_idx is not None
        for req in batch.reqs
    )


def try_process_ready_prefill_result(scheduler, current_batch):
    """Process the queue head once, only after its recorded copy event is ready.

    Unsupported/not-ready inputs leave all state unchanged. Queue invariants and
    device/API/processing errors propagate rather than being hidden as fallback.
    """
    if (
        current_batch is None
        or scheduler.last_batch is None
        or scheduler.chunked_req is not None
    ):
        return False
    previous_batch, result = scheduler.result_queue[0]
    if not (
        _ordinary_final_requests(previous_batch)
        and _ordinary_final_requests(current_batch)
    ):
        return False
    previous_rows = {req.kv.req_pool_idx for req in previous_batch.reqs}
    current_rows = {req.kv.req_pool_idx for req in current_batch.reqs}
    if (
        len(previous_rows) != len(previous_batch.reqs)
        or len(current_rows) != len(current_batch.reqs)
        or previous_rows & current_rows
        or {id(req) for req in previous_batch.reqs}
        & {id(req) for req in current_batch.reqs}
    ):
        return False
    if (
        not result.copy_done_recorded
        or result.copy_done is None
        or result.delay_sample_func is not None
        or not result.has_sampled_token_ids
        or result.next_token_ids.device.type != "cpu"
        or result.next_draft_input is not None
        or result.can_run_cuda_graph
        or result.routed_experts_output is not None
        or result.indexer_topk_output is not None
        or result.expert_distribution_metrics is not None
        or result.auxiliary_host_output is not None
    ):
        return False
    if not result.copy_done.query():
        return False
    scheduler.result_queue.popleft()
    # Keep the complete existing processing path, radix pinning and tensor refs.
    # The existing later transfer poll owns all KV/metadata release operations.
    scheduler.process_batch_result(previous_batch, result)
    return True
