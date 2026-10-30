"""Dense MLA P-cache tests. Set P_KV_TEST_DEVICE=npu:0 for hardware checks."""

import json
import os
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.hardware_backend.npu.attention.prefill_expanded_kv import (
    ExpandedKVGeometry,
    expanded_kv_geometry,
    forward_expanded_prefill,
)


def test_geometry_and_capacity():
    config = SimpleNamespace(
        num_attention_heads=20, qk_nope_head_dim=192, qk_rope_head_dim=64, v_head_dim=256
    )
    geometry = expanded_kv_geometry(config, 1)
    assert geometry == ExpandedKVGeometry(20, 256, 256)
    # Both the PD latent cache and local expanded cache are required.
    assert (geometry.width + 576) * 47 * 2 == 1_016_704
    assert expanded_kv_geometry(config, 2).width == geometry.width // 2
    with pytest.raises(ValueError, match="divisible"):
        expanded_kv_geometry(config, 3)


def make_pool(monkeypatch, device, layers=2, pages=40):
    import sglang.srt.hardware_backend.npu.memory_pool_npu as pools

    monkeypatch.setattr(pools, "use_unified_mla_kv_layout", lambda: True)
    return pools.NPUMLATokenToKVPool(
        size=(pages - 1) * 128,
        page_size=128,
        dtype=torch.bfloat16,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        layer_num=layers,
        device=device,
        enable_memory_saver=False,
        start_layer=3,
        end_layer=3 + layers,
        prefill_expanded_geometry=ExpandedKVGeometry(20, 256, 256),
    )


def test_pd_registration_stays_latent(monkeypatch):
    pool = make_pool(monkeypatch, "cpu", pages=2)
    pointers, lengths, items = pool.get_contiguous_buf_infos()
    assert len(pointers) == pool.layer_num == 2
    assert items == [128 * 576 * 2] * 2
    assert lengths == [2 * items[0]] * 2
    expanded_ptrs = {
        t.data_ptr() for t in pool.prefill_k_buffer + pool.prefill_v_buffer
    }
    assert not (set(pointers) & expanded_ptrs)
    assert pool.get_kv_size_bytes() == 2 * 2 * 128 * (576 + 10240) * 2
    assert pool.get_prefill_expanded_kv(4)[0].is_contiguous()


def test_pool_sizing_includes_expanded(monkeypatch):
    import sglang.srt.model_executor.pool_configurator as sizing
    import sglang.srt.layers.cp.utils as cp_utils
    from sglang.srt.environ import envs

    config = SimpleNamespace(
        num_attention_heads=20, qk_nope_head_dim=192, qk_rope_head_dim=64,
        kv_lora_rank=512, v_head_dim=256,
        hf_config=SimpleNamespace(architectures=["Glm4MoeLiteForCausalLM"]),
    )
    parallel = SimpleNamespace(
        attn_tp_size=1, attn_dcp_size=1, attn_cp_size=1,
        enable_dsa_cache_layer_split=False,
    )
    monkeypatch.setattr(sizing, "get_parallel", lambda: parallel)
    monkeypatch.setattr(cp_utils, "get_parallel", lambda: parallel)
    monkeypatch.setattr(sizing, "get_disagg", lambda: SimpleNamespace(disaggregation_mode="prefill"))
    monkeypatch.setattr(sizing, "get_memory", lambda: SimpleNamespace(enable_hierarchical_cache=False))
    kvc = SimpleNamespace(
        model_config=config, kv_cache_dtype=torch.bfloat16, use_mla_backend=True,
        device="npu", spec_algorithm=SimpleNamespace(is_none=lambda: True),
        is_draft_worker=False,
    )
    configurator = sizing.DefaultPoolConfigurator.__new__(sizing.DefaultPoolConfigurator)
    monkeypatch.setenv("SGLANG_NPU_PREFILL_EXPANDED_KV", "1")
    assert envs.SGLANG_NPU_PREFILL_EXPANDED_KV.get()
    assert configurator._compute_cell_size(kvc, 47) == 1_016_704
    configurator._cell_size = 1_016_704
    configurator._expanded_prefill_kv = True
    budget = (1024 + 128) * configurator._cell_size
    assert configurator.calculate_pool_sizes(budget, 128).max_total_num_tokens == 1024
    monkeypatch.setenv("SGLANG_NPU_PREFILL_EXPANDED_KV", "0")
    assert configurator._compute_cell_size(kvc, 47) == 54_144


@pytest.mark.skipif(not os.environ.get("P_KV_TEST_DEVICE"), reason="NPU opt-in")
@pytest.mark.parametrize(
    "prefix_lens,query_lens,padding,shared",
    [([0], [17], 0, False), ([127], [129], 0, False),
     ([384, 256], [17, 31], 0, False), ([2051], [16], 0, False),
     ([128, 129], [0, 17], 0, False), ([129], [17], 7, False),
     ([256, 256], [17, 31], 0, True)],
)
def test_paged_prefix_against_fp32(monkeypatch, prefix_lens, query_lens, padding, shared):
    import torch_npu

    torch.set_num_threads(4)
    torch.manual_seed(42)
    device = os.environ["P_KV_TEST_DEVICE"]
    torch.npu.set_device(device)
    pool = make_pool(monkeypatch, device, layers=1)
    key_cache, value_cache = pool.get_prefill_expanded_kv(3)
    key_cache.fill_(float("nan"))
    value_cache.fill_(float("nan"))
    latent_before = pool.kv_buffer.clone()
    max_pages = max((p + q + 127) // 128 for p, q in zip(prefix_lens, query_lens))
    table = torch.zeros(len(prefix_lens), max_pages, dtype=torch.int32)
    permutation = torch.randperm(39) + 1  # Leave dummy page zero unused.
    page_offset = 0
    old_keys, old_values, old_slots = [], [], []
    new_keys, new_values, new_slots, queries, golden = [], [], [], [], []
    cumulative = []
    for req, (prefix, count) in enumerate(zip(prefix_lens, query_lens)):
        length = prefix + count
        page_count = (length + 127) // 128
        pages = permutation[page_offset : page_offset + page_count]
        page_offset += page_count
        if shared and req:
            pages = pages.clone()
            pages[:prefix // 128] = table[0, :prefix // 128]
        table[req, :page_count] = pages
        slots = (pages[:, None] * 128 + torch.arange(128)).flatten()[:length]
        key = (torch.randn(length, 20, 256) * 0.2).bfloat16()
        value = torch.randn(length, 20, 256).bfloat16()
        if shared and req:
            key[:prefix] = old_keys[0]
            value[:prefix] = old_values[0]
        query = (torch.randn(count, 20, 256) * 0.2).bfloat16()
        old_slots.append(slots[:prefix])
        old_keys.append(key[:prefix])
        old_values.append(value[:prefix])
        new_slots.append(slots[prefix:])
        new_keys.append(key[prefix:])
        new_values.append(value[prefix:])
        queries.append(query)
        cumulative.append(sum(query_lens[: req + 1]))
        if count:
            scores = torch.einsum("qhd,khd->hqk", query.float(), key.float()) / 16
            mask = torch.arange(length)[None, :] > prefix + torch.arange(count)[:, None]
            scores.masked_fill_(mask[None], float("-inf"))
            golden.append(torch.einsum("hqk,khd->qhd", scores.softmax(-1), value.float()))
    pool.set_prefill_expanded_kv(
        3, torch.cat(old_slots).to(device), torch.cat(old_keys).to(device),
        torch.cat(old_values).to(device),
    )
    metadata = SimpleNamespace(
        block_tables=table.to(device), seq_lens_list_cumsum=cumulative
    )
    backend = SimpleNamespace(
        token_to_kv_pool=pool, forward_metadata=metadata,
        fia_mask=torch.triu(torch.ones(2048, 2048, dtype=torch.bool), diagonal=1).to(device),
        _get_seq_lens_cpu_int_list=lambda: [p + q for p, q in zip(prefix_lens, query_lens)],
    )
    layer = SimpleNamespace(
        layer_id=3, tp_k_head_num=20, tp_q_head_num=20,
        qk_head_dim=256, v_head_dim=256, scaling=1 / 16,
    )
    query = torch.cat(queries)
    if padding:
        query = torch.cat([query, torch.zeros(padding, 20, 256, dtype=query.dtype)])
    batch = SimpleNamespace(
        num_token_non_padded_cpu=sum(query_lens),
        out_cache_loc=torch.cat(new_slots).to(device),
        extend_prefix_lens_cpu=prefix_lens,
    )
    actual = forward_expanded_prefill(
        backend, query.to(device), torch.cat(new_keys).to(device),
        torch.cat(new_values).to(device), layer, batch,
    ).cpu().float()
    torch.npu.synchronize()
    expected = torch.cat(golden).flatten(1)
    actual_valid = actual[: len(expected)]
    # FP64 reductions avoid float32 cosine reduction error over large outputs.
    actual64, expected64 = actual_valid.double(), expected.double()
    relative_l2 = (actual64 - expected64).norm() / expected64.norm()
    cosine = torch.nn.functional.cosine_similarity(actual64.flatten(), expected64.flatten(), dim=0)
    print(json.dumps(dict(prefix=prefix_lens, query=query_lens, padding=padding, shared=shared,
                         relative_l2=relative_l2.item(), cosine=cosine.item())))
    assert torch.isfinite(actual).all(), "Read an unwritten cache slot or incorrect mask"
    assert relative_l2 < 0.005, relative_l2.item()
    assert cosine > 0.99999, cosine.item()
    assert torch.equal(pool.kv_buffer, latent_before), "Expanded writes changed the PD cache"
    written_slots = torch.cat(new_slots).long()
    assert torch.equal(key_cache.cpu().flatten(0, 1)[written_slots], torch.cat(new_keys))
    assert torch.equal(value_cache.cpu().flatten(0, 1)[written_slots], torch.cat(new_values))
    if padding:
        assert actual[len(expected):].count_nonzero() == 0
