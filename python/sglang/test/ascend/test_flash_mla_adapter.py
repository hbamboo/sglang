"""Unit checks for the FlashMLA adapter + NPU MLA single-buffer pool.

ENABLE_ASCEND_FLASH_MLA must be on (the test sets it if unset).

Run inside the container (card 0):

    source /mnt/sfs_turbo/hxd/pkgs/flash_mla_delivery/env.sh   # maybe already in env
    ASCEND_RT_VISIBLE_DEVICES=0 python3 \
        /mnt/sfs_turbo/hxd/flash_mla_work/test_flash_mla_npu.py

Checks
  1. adapter.decode layout: paged PA_BBND latent (576) -> torch fp32 reference
  2. NPUMLATokenToKVPool.set_kv_buffer writes (latent || rope) interleaved
  3. metadata reuse: values-only change gives bit-identical output
"""

import math
import os
import sys

os.environ.setdefault("ENABLE_ASCEND_FLASH_MLA", "1")

import torch
import torch_npu  # noqa: F401

from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.hardware_backend.npu.attention import (  # noqa: E402
    flash_mla_adapter as fmla,
)

DEV = "npu:0"
PAGE = 128
HEAD_DIM_QK = 576
HEAD_DIM_V = 512
H = 16


def err_stats(out, ref):
    out = out.float()
    ref = ref.float()
    cos = torch.nn.functional.cosine_similarity(
        out.flatten(), ref.flatten(), dim=0
    ).item()
    diff = (out - ref).abs()
    rms = ref.pow(2).mean().sqrt().item()
    return {
        "cos": cos,
        "max_abs": diff.max().item(),
        "mean_abs_over_rms": diff.mean().item() / rms,
        "max_abs_over_rms": diff.max().item() / rms,
    }


def gather_kv(k_cache, block_table, seqlen):
    pages = (seqlen + PAGE - 1) // PAGE
    blocks = block_table[:pages].long()
    kv = k_cache.index_select(0, blocks).reshape(-1, HEAD_DIM_QK)
    return kv[:seqlen]


def ref_decode(q, k_cache, block_table, seqlens, scale):
    outs = []
    for b, sl in enumerate(seqlens):
        kv = gather_kv(k_cache, block_table[b], sl).float()
        k = kv
        v = kv[:, :HEAD_DIM_V]
        logits = (q[b, 0].float() @ k.T) * scale
        p = torch.softmax(logits, dim=-1)
        outs.append(p @ v)
    return torch.stack(outs).unsqueeze(1)


def test_adapter_decode():
    torch.manual_seed(0)
    seqs = [1024, 777, 3014, 1]
    bs = len(seqs)
    nblocks = 128
    k_cache = torch.randn(
        nblocks, PAGE, 1, HEAD_DIM_QK, dtype=torch.bfloat16, device=DEV
    )
    q = torch.randn(bs, 1, H, HEAD_DIM_QK, dtype=torch.bfloat16, device=DEV)
    max_pages = max((s + PAGE - 1) // PAGE for s in seqs)
    perm = torch.randperm(nblocks, dtype=torch.int32, device=DEV)
    bt = perm[: bs * max_pages].reshape(bs, max_pages)
    cs = torch.tensor(seqs, dtype=torch.int32, device=DEV)
    scale = 1.0 / math.sqrt(HEAD_DIM_QK)

    out = fmla.decode(
        q,
        k_cache,
        bt,
        cs,
        num_q_heads=H,
        scale=scale,
        max_seqlen_kv=3014,
    )
    torch.npu.synchronize()
    ref = ref_decode(q, k_cache, bt, seqs, scale)
    return out, ref


def test_pool_write():
    from sglang.srt.hardware_backend.npu import memory_pool_npu as mp

    assert mp.is_flash_mla_enabled()

    pool = mp.NPUMLATokenToKVPool.__new__(mp.NPUMLATokenToKVPool)
    pool.size = 4 * PAGE
    pool.page_size = PAGE
    pool.dtype = torch.bfloat16
    pool.store_dtype = torch.bfloat16
    pool.start_layer = 0
    pool.layer_num = 2
    pool.layer_transfer_counter = None
    pool.kv_lora_rank = HEAD_DIM_V
    pool.qk_rope_head_dim = HEAD_DIM_QK - HEAD_DIM_V
    pool.index_head_dim = None
    pool.flash_mla = True
    pool.kv_buffer = torch.zeros(
        (2, 5, PAGE, 1, HEAD_DIM_QK), dtype=torch.bfloat16, device=DEV
    )
    pool.k_buffer = pool.kv_buffer[..., : pool.kv_lora_rank]
    pool.v_buffer = pool.kv_buffer[..., pool.kv_lora_rank :]

    class L:
        layer_id = 0

    loc = torch.tensor([3, 1, 4], dtype=torch.int64, device=DEV)
    k = torch.randn(3, 1, HEAD_DIM_V, dtype=torch.bfloat16, device=DEV)
    v = torch.randn(
        3, 1, HEAD_DIM_QK - HEAD_DIM_V, dtype=torch.bfloat16, device=DEV
    )
    pool.set_kv_buffer(L(), loc, k, v)
    torch.npu.synchronize()

    flat = pool.kv_buffer[0].reshape(-1, HEAD_DIM_QK)
    got = flat[loc.cpu()]
    want = torch.cat([k.reshape(3, -1), v.reshape(3, -1)], dim=-1)
    ok = torch.equal(got, want)
    # the returned view must be the same PA_BBND buffer
    ok = ok and pool.get_flash_mla_kv_buffer(0).data_ptr() == pool.kv_buffer[
        0
    ].data_ptr()
    return ok


def test_metadata_reuse():
    torch.manual_seed(1)
    bs = 16
    seqs = [3014] * bs
    nblocks = 512
    k_cache = torch.randn(
        nblocks, PAGE, 1, HEAD_DIM_QK, dtype=torch.bfloat16, device=DEV
    )
    q = torch.randn(bs, 1, H, HEAD_DIM_QK, dtype=torch.bfloat16, device=DEV)
    max_pages = (3014 + PAGE - 1) // PAGE
    bt = torch.randint(
        0, nblocks, (bs, max_pages), dtype=torch.int32, device=DEV
    )
    cs = torch.tensor(seqs, dtype=torch.int32, device=DEV)
    scale = 1.0 / math.sqrt(HEAD_DIM_QK)

    md = fmla.metadata(
        H, batch_size=bs, max_seqlen_q=1, max_seqlen_kv=3014, cache_seqlens=cs
    )
    torch.npu.synchronize()

    new_seqs = [2048 + (i * 61) % 900 for i in range(bs)]
    new_seqs[0] = 3014
    cs2 = torch.tensor(new_seqs, dtype=torch.int32, device=DEV)

    out_stale = fmla.decode(
        q, k_cache, bt, cs2, num_q_heads=H, scale=scale, max_seqlen_kv=3014, md=md
    )
    md2 = fmla.metadata(
        H, batch_size=bs, max_seqlen_q=1, max_seqlen_kv=3014, cache_seqlens=cs2
    )
    out_fresh = fmla.decode(
        q, k_cache, bt, cs2, num_q_heads=H, scale=scale, max_seqlen_kv=3014, md=md2
    )
    ref = ref_decode(q, k_cache, bt, new_seqs, scale)
    torch.npu.synchronize()
    return out_stale, out_fresh, ref


def ref_extend(q, k_cache, block_table, prefix_lens, q_lens):
    """Right-aligned causal reference for the TND extend path."""
    outs = []
    off = 0
    for b, (p, l) in enumerate(zip(prefix_lens, q_lens)):
        total = p + l
        pages = (total + PAGE - 1) // PAGE
        blocks = block_table[b, :pages].long()
        kv = k_cache.index_select(0, blocks).reshape(-1, HEAD_DIM_QK)[:total].float()
        for i in range(l):
            k = kv[: p + i + 1]
            v = k[:, :HEAD_DIM_V]
            pr = torch.softmax((q[off + i].float() @ k.T) * (1.0 / math.sqrt(HEAD_DIM_QK)), -1)
            outs.append(pr @ v)
        off += l
    return torch.stack(outs)


def test_adapter_extend():
    """TND prefill: prefix + chunk, ragged batch, causal."""
    torch.manual_seed(3)
    q_lens = [64, 1, 200]
    prefix_lens = [768, 5, 3000]
    total = [p + l for p, l in zip(prefix_lens, q_lens)]
    nb = 64
    k_cache = torch.randn(
        nb, PAGE, 1, HEAD_DIM_QK, dtype=torch.bfloat16, device=DEV
    )
    q = torch.randn(
        sum(q_lens), H, HEAD_DIM_QK, dtype=torch.bfloat16, device=DEV
    )
    max_pages = max((t + PAGE - 1) // PAGE for t in total)
    bt = torch.randint(0, nb, (len(q_lens), max_pages), dtype=torch.int32, device=DEV)
    cs = torch.tensor(total, dtype=torch.int32, device=DEV)
    cu = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(q_lens), 0).tolist()),
        dtype=torch.int32,
        device=DEV,
    )
    out = fmla.extend(
        q,
        k_cache,
        bt,
        cs,
        cu,
        num_q_heads=H,
        scale=1.0 / math.sqrt(HEAD_DIM_QK),
        max_seqlen_q=max(q_lens),
        max_seqlen_kv=max(total),
    )
    torch.npu.synchronize()
    ref = ref_extend(q, k_cache, bt, prefix_lens, q_lens)
    return out, ref


if __name__ == "__main__":
    print("device:", torch_npu.npu.get_device_name(0))
    out, ref = test_adapter_decode()
    print("1) decode vs fp32 ref:", err_stats(out, ref))

    ok = test_pool_write()
    print("2) pool set_kv_buffer interleaved:", "OK" if ok else "FAIL")
    assert ok

    a, b, ref2 = test_metadata_reuse()
    print("3) stale vs fresh metadata max_abs:", (a - b).abs().max().item())
    print("   stale vs fp32 ref:", err_stats(a, ref2))
    print("   fresh vs fp32 ref:", err_stats(b, ref2))

    out_e, ref_e = test_adapter_extend()
    print("4) TND extend vs fp32 ref:", err_stats(out_e, ref_e))
