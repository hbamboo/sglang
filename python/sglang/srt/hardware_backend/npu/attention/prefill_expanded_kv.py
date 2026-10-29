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


def expanded_kv_geometry(model_config, tp_size):
    heads = model_config.num_attention_heads
    if heads % tp_size:
        raise ValueError("Expanded prefill KV requires query heads divisible by TP")
    key_dim = model_config.qk_nope_head_dim + model_config.qk_rope_head_dim
    value_dim = model_config.v_head_dim
    if key_dim != value_dim:
        raise ValueError("Expanded prefill KV currently requires equal K/V head dims")
    return ExpandedKVGeometry(heads // tp_size, key_dim, value_dim)


def forward_expanded_prefill(backend, q, k, v, layer, forward_batch):
    """Write only new rows, then let paged FIA read prefix and chunk in place."""
    import torch
    import torch_npu

    pool = backend.token_to_kv_pool
    geometry = pool.prefill_expanded_geometry
    if (
        k is None
        or v is None
        or layer.tp_k_head_num != geometry.heads
        or layer.qk_head_dim != geometry.key_dim
        or layer.v_head_dim != geometry.value_dim
    ):
        raise ValueError("Expanded prefill KV requires the dense MHA prefill route")
    tokens = forward_batch.num_token_non_padded_cpu
    query = q.reshape(-1, layer.tp_q_head_num, layer.qk_head_dim)
    if tokens == 0:
        return query.new_zeros((query.shape[0], layer.tp_q_head_num * layer.v_head_dim))
    pool.set_prefill_expanded_kv(
        layer.layer_id,
        forward_batch.out_cache_loc[:tokens],
        k[:tokens],
        v[:tokens],
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
