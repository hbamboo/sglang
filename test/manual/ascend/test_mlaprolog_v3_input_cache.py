"""Forward-scoped RoPE sharing and graph-safe scratch index regressions.

Run manually on an explicitly selected NPU with pytest. CPU cases cover the actual
GLM model entry so reusing a ForwardBatch cannot keep stale RoPE outputs.
"""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.hardware_backend.npu.attention import mla_preprocess as prolog


def make_layer(rope):
    return SimpleNamespace(rotary_emb=rope, kv_lora_rank=512, qk_rope_head_dim=64)


def get_rope(layer, positions, batch=None):
    return prolog.NPUFusedMLAPreprocess.get_sin_cos(layer, positions, batch)


def test_batch_shares_only_matching_rope_and_positions():
    rope = SimpleNamespace(cos_sin_cache=torch.randn(128, 64))
    first, second = make_layer(rope), make_layer(rope)
    positions = torch.tensor([1, 7, 23, 2])
    batch = SimpleNamespace(npu_mlaprolog_rope_cache={})
    expected = get_rope(first, positions)
    actual = get_rope(first, positions, batch)
    shared = get_rope(second, positions, batch)
    for a, b, c in zip(expected, actual, shared):
        assert torch.equal(a, b)
        assert b is c
        assert b.is_contiguous()
    # TBO subbatches/views and different RoPE configurations never collide.
    other = get_rope(second, positions.flip(0), batch)
    assert torch.equal(other[0], get_rope(first, positions.flip(0))[0])
    assert other[0] is not actual[0]
    different_rope = make_layer(SimpleNamespace(cos_sin_cache=rope.cos_sin_cache + 1))
    other = get_rope(different_rope, positions, batch)
    assert torch.equal(other[0], get_rope(different_rope, positions)[0])
    assert other[0] is not actual[0]


@pytest.mark.parametrize("use_child", [False, True])
def test_glm_entry_refreshes_reused_batch(monkeypatch, use_child):
    from sglang.srt.models import glm4_moe_lite as glm

    monkeypatch.setattr(glm, "_is_npu", True)
    monkeypatch.setattr(glm, "BumpAllocator", lambda **kwargs: None)
    recorder = SimpleNamespace(with_current_layer=lambda _: nullcontext())
    monkeypatch.setattr(
        glm, "get_global_expert_distribution_recorder", lambda: recorder
    )
    rope = SimpleNamespace(cos_sin_cache=torch.randn(128, 64))
    layer = make_layer(rope)
    seen = []
    child = SimpleNamespace(npu_mlaprolog_rope_cache={})

    def run_layer(positions, hidden, batch, residual, allocator):
        seen.append(get_rope(layer, positions, child if use_child else batch))
        return hidden, residual

    # Nonzero first layer also covers a PP-stage-like layer range.
    model = SimpleNamespace(
        start_layer=4,
        end_layer=5,
        pp_group=SimpleNamespace(is_first_rank=True, is_last_rank=True),
        embed_tokens=lambda _: torch.ones(4, 8),
        layers={4: run_layer},
        layers_to_capture=[],
        norm=lambda x: x,
    )
    batch = SimpleNamespace(
        can_run_tbo=False,
        tbo_children=[child],
        forward_mode=SimpleNamespace(is_idle=lambda: False),
    )
    positions = torch.tensor([1, 7, 23, 2])
    ids = torch.zeros(4, dtype=torch.int64)
    glm.Glm4MoeLiteModel.forward(model, ids, positions, batch)
    positions.add_(1)
    glm.Glm4MoeLiteModel.forward(model, ids, positions, batch)
    assert seen[0][0] is not seen[1][0]
    assert not torch.equal(seen[0][0], seen[1][0])
    assert torch.equal(seen[1][0], get_rope(layer, positions)[0])


@pytest.mark.parametrize("tokens", [0, 1, 8, 12, 13, 16, 17, 129])
def test_scratch_index_is_contiguous_and_reused(tokens):
    layer = make_layer(None)
    pool = SimpleNamespace(
        store_dtype=torch.bfloat16, dtype=torch.bfloat16, page_size=128
    )
    x = torch.empty(tokens, 2048, dtype=torch.bfloat16)
    scratch = prolog.NPUFusedMLAPreprocess._v3_scratch_cache
    _, _, first = scratch(layer, x, pool)
    _, _, second = scratch(layer, x, pool)
    assert first.dtype == torch.int64 and first.is_contiguous()
    assert torch.equal(first, torch.arange(tokens))
    assert first.data_ptr() == second.data_ptr()


def test_growing_index_keeps_old_graph_storage_alive():
    layer = make_layer(None)
    pool = SimpleNamespace(
        store_dtype=torch.bfloat16, dtype=torch.bfloat16, page_size=128
    )
    scratch = prolog.NPUFusedMLAPreprocess._v3_scratch_cache
    _, _, old = scratch(layer, torch.empty(12, 2048), pool)
    old_ptr = old.data_ptr()
    scratch(layer, torch.empty(129, 2048), pool)
    _, _, old_again = scratch(layer, torch.empty(12, 2048), pool)
    assert old_ptr == old_again.data_ptr()
    assert torch.equal(old, torch.arange(12))


def test_tbo_split_does_not_inherit_parent_rope_results(monkeypatch):
    from sglang.srt.batch_overlap import two_batch_overlap as tbo
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

    monkeypatch.setattr(
        tbo,
        "get_parallel",
        lambda: SimpleNamespace(attn_tp_size=1, moe_dense_tp_size=1),
    )
    batch = ForwardBatch(
        forward_mode=ForwardMode.DECODE,
        batch_size=4,
        input_ids=torch.zeros(4, dtype=torch.int64),
        req_pool_indices=torch.arange(4),
        seq_lens=torch.ones(4, dtype=torch.int64),
        out_cache_loc=torch.arange(4),
        seq_lens_sum=4,
        positions=torch.arange(4),
        npu_mlaprolog_rope_cache={"previous-forward": (torch.ones(4, 64),)},
    )
    child = tbo.TboForwardBatchPreparer.filter_batch(
        batch,
        start_token_index=0,
        end_token_index=2,
        start_seq_index=0,
        end_seq_index=2,
        out_num_token_non_padded=torch.tensor(2),
    )
    assert child.npu_mlaprolog_rope_cache is None
    assert child.positions.shape == (2,)


@pytest.mark.parametrize("tokens", [8, 12, 16])
def test_npu_graph_refreshes_positions(tokens):
    import torch_npu  # noqa: F401 - register the NPU backend before graph capture

    device = "npu:0"
    rope = SimpleNamespace(
        cos_sin_cache=torch.randn(4096, 64, device=device, dtype=torch.bfloat16)
    )
    first, second = make_layer(rope), make_layer(rope)
    batch = SimpleNamespace(npu_mlaprolog_rope_cache={})
    positions = torch.arange(tokens, device=device, dtype=torch.int64)
    x = torch.empty(tokens, 2048, device=device, dtype=torch.bfloat16)
    pool = SimpleNamespace(store_dtype=x.dtype, dtype=x.dtype, page_size=128)

    def forward():
        # The production model entry performs this reset for each warmup/capture.
        batch.npu_mlaprolog_rope_cache = {}
        a = get_rope(first, positions, batch)
        b = get_rope(second, positions, batch)
        assert a[0] is b[0] and a[1] is b[1]
        _, _, idx = prolog.NPUFusedMLAPreprocess._v3_scratch_cache(first, x, pool)
        return a, idx

    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    graph = torch.npu.NPUGraph()
    with torch.npu.stream(stream):
        for _ in range(3):
            forward()
        stream.synchronize()
        with torch.npu.graph(graph, stream=stream):
            captured, idx = forward()
    torch.npu.current_stream().wait_stream(stream)
    for offset in [3, 31, 511]:
        positions.copy_(torch.arange(tokens, device=device) + offset)
        graph.replay()
        torch.npu.synchronize()
        expected = get_rope(first, positions)
        assert torch.equal(captured[0], expected[0])
        assert torch.equal(captured[1], expected[1])
        assert torch.equal(idx.cpu(), torch.arange(tokens))
