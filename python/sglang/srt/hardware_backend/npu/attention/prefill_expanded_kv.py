"""Paged expanded KV for dense MLA prefillers; latent KV remains the PD ABI."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ExpandedKVGeometry:
    heads: int
    key_dim: int
    value_dim: int

    @property
    def width(self):
        return self.heads * (self.key_dim + self.value_dim)


@dataclass(frozen=True)
class PackedExpandedKV:
    """Per-layer inputs carried explicitly from prepare to attention core."""
    projection: object
    k_rope: object
    nope_dim: int


def can_fuse_expanded_kv(pool, forward_batch, projection, k_rope):
    import torch
    from sglang.srt.environ import envs
    from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
        get_tc_piecewise_forward_context,
    )

    geometry = getattr(pool, "prefill_expanded_geometry", None)
    return (
        envs.SGLANG_NPU_PREFILL_FUSED_KV_SCATTER.get()
        and geometry is not None
        and forward_batch.forward_mode.is_extend()
        and any(forward_batch.extend_prefix_lens_cpu)
        and not torch.compiler.is_compiling()
        and get_tc_piecewise_forward_context() is None
        and projection.dtype in (torch.bfloat16, torch.float16)
        and projection.dtype == k_rope.dtype
        and projection.is_contiguous()
        and k_rope.is_contiguous()
    )


def expanded_kv_geometry(model_config, tp_size):
    heads = model_config.num_attention_heads
    if heads % tp_size:
        raise ValueError("Expanded prefill KV requires query heads divisible by TP")
    key_dim = model_config.qk_nope_head_dim + model_config.qk_rope_head_dim
    value_dim = model_config.v_head_dim
    if key_dim != value_dim:
        raise ValueError("Expanded prefill KV currently requires equal K/V head dims")
    return ExpandedKVGeometry(heads // tp_size, key_dim, value_dim)


def get_prefill_expanded_geometry(kvc):
    """Auto-enable supported P workers; keep explicit opt-in validation strict."""
    import torch

    from sglang.srt.configs.model_config import is_deepseek_dsa
    from sglang.srt.environ import envs
    from sglang.srt.runtime_context import get_disagg, get_memory, get_parallel

    flag = envs.SGLANG_NPU_PREFILL_EXPANDED_KV
    if not flag.get():
        return None
    supported = (
        str(kvc.device).startswith("npu")
        and kvc.use_mla_backend
        and get_disagg().disaggregation_mode == "prefill"
        and not is_deepseek_dsa(kvc.model_config.hf_config)
        and kvc.kv_cache_dtype in (torch.bfloat16, torch.float16)
        and get_parallel().attn_dcp_size == 1
        and get_parallel().attn_cp_size == 1
        and kvc.spec_algorithm.is_none()
        and not get_memory().enable_hierarchical_cache
    )
    if not supported:
        if flag.is_set():
            raise ValueError(
                "SGLANG_NPU_PREFILL_EXPANDED_KV requires dense NPU MLA PD prefill, "
                "BF16/FP16 KV, CP=1, and no speculation or hierarchical cache"
            )
        return None
    try:
        return expanded_kv_geometry(kvc.model_config, get_parallel().attn_tp_size)
    except ValueError:
        if flag.is_set():
            raise
        return None


def forward_expanded_prefill(backend, q, k, v, layer, forward_batch, *, packed_kv=None):
    """Write only new rows, then let paged FIA read prefix and chunk in place."""
    import torch
    import torch_npu

    pool = backend.token_to_kv_pool
    geometry = pool.prefill_expanded_geometry
    if (
        (packed_kv is None and (k is None or v is None))
        or layer.tp_k_head_num != geometry.heads
        or layer.qk_head_dim != geometry.key_dim
        or layer.v_head_dim != geometry.value_dim
    ):
        raise ValueError("Expanded prefill KV requires the dense MHA prefill route")
    tokens = forward_batch.num_token_non_padded_cpu
    query = q.reshape(-1, layer.tp_q_head_num, layer.qk_head_dim)
    if tokens == 0:
        return query.new_zeros((query.shape[0], layer.tp_q_head_num * layer.v_head_dim))
    if packed_kv is None:
        pool.set_prefill_expanded_kv(
            layer.layer_id,
            forward_batch.out_cache_loc[:tokens],
            k[:tokens],
            v[:tokens],
        )
    else:
        from sgl_kernel_npu.scatter_pa_kv_cache import scatter_pa_kv_cache_mla

        if not any(forward_batch.extend_prefix_lens_cpu):
            raise ValueError("Packed expanded KV requires the paged prefix path")
        key_cache, value_cache = pool.get_prefill_expanded_kv(layer.layer_id)
        scatter_pa_kv_cache_mla(
            packed_kv.projection[:tokens],
            packed_kv.k_rope[:tokens],
            forward_batch.out_cache_loc[:tokens],
            key_cache,
            value_cache,
            packed_kv.nope_dim,
        )
    metadata = backend.forward_metadata
    if not any(forward_batch.extend_prefix_lens_cpu):
        # The whole K/V sequence already exists as contiguous projection
        # outputs. Keep caching it for future hits, but avoid paged-attention
        # overhead when no history needs to be read.
        output, _ = torch.ops.npu.npu_fused_infer_attention_score(
            query[:tokens],
            k[:tokens].reshape(tokens, geometry.heads, geometry.key_dim).contiguous(),
            v[:tokens].reshape(tokens, geometry.heads, geometry.value_dim).contiguous(),
            num_heads=layer.tp_q_head_num,
            num_key_value_heads=geometry.heads,
            input_layout="TND",
            actual_seq_lengths=metadata.seq_lens_list_cumsum,
            actual_seq_lengths_kv=metadata.seq_lens_list_cumsum,
            atten_mask=backend.fia_mask,
            sparse_mode=3,
            next_tokens=0,
            scale=layer.scaling,
        )
    else:
        key, value = pool.get_prefill_expanded_kv(layer.layer_id)
        output, _ = torch_npu.npu_fused_infer_attention_score_v2(
            query=query[:tokens],
            key=key.view(-1, pool.page_size, geometry.heads * geometry.key_dim),
            value=value.view(-1, pool.page_size, geometry.heads * geometry.value_dim),
            block_table=metadata.block_tables,
            block_size=pool.page_size,
            input_layout="TND",
            num_query_heads=layer.tp_q_head_num,
            num_key_value_heads=geometry.heads,
            actual_seq_qlen=metadata.seq_lens_list_cumsum,
            actual_seq_kvlen=backend._get_seq_lens_cpu_int_list(),
            atten_mask=backend.fia_mask,
            sparse_mode=3,
            pre_tokens=2147483647,
            next_tokens=0,
            softmax_scale=layer.scaling,
        )
    output = output.reshape(tokens, layer.tp_q_head_num * layer.v_head_dim)
    if query.shape[0] != tokens:
        padded = torch.zeros(
            (query.shape[0], output.shape[1]), dtype=output.dtype, device=output.device
        )
        padded[:tokens].copy_(output)
        output = padded
    return output
