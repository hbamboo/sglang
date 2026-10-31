"""V3 decode writes a full Query and FlashMLA consumes it without concatenation."""
from types import SimpleNamespace
import types

import pytest
import torch
import torch_npu
from sgl_kernel_npu.mla_prolog_v3_compat import direct
from sglang.srt.hardware_backend.npu.attention import mla_preprocess as prolog
from sglang.srt.hardware_backend.npu.attention import flash_mla_adapter as adapter
from sglang.srt.hardware_backend.npu.attention.ascend_backend import AscendAttnBackend
from sglang.srt.hardware_backend.npu import memory_pool_npu as pools

@pytest.mark.parametrize('tokens',[1,7,32,64,92])
@pytest.mark.parametrize('direct_cache',[False,True])
def test_wrapper_to_flashmla(monkeypatch,tokens,direct_cache):
    torch.manual_seed(42)
    def rand(*shape):return (torch.randn(*shape)*.02).to(dtype=torch.bfloat16,device='npu:0')
    def nz(*shape):return torch_npu.npu_format_cast(rand(*shape),29)
    monkeypatch.setattr(pools,'use_unified_mla_kv_layout',lambda:True)
    pool=pools.NPUMLATokenToKVPool(size=2048,page_size=128,dtype=torch.bfloat16,
        kv_lora_rank=512,qk_rope_head_dim=64,layer_num=1,device='npu:0',enable_memory_saver=False)
    pool.kv_buffer.fill_(123)
    idx=torch.arange(tokens,device='npu:0',dtype=torch.int64)*17%2048
    hidden=rand(tokens,2048);sin=rand(tokens,64);cos=rand(tokens,64)
    obj=SimpleNamespace(has_preprocess_weights=True,layer_id=0,kv_lora_rank=512,
        qk_rope_head_dim=64,num_local_heads=20,q_lora_rank=768,
        weight_dq=nz(2048,768),weight_uq_qr=nz(768,5120),weight_uk=rand(20,192,512),
        weight_dkv_kr=nz(2048,576),q_a_layernorm=SimpleNamespace(weight=rand(768)+1,variance_epsilon=1e-5),
        kv_a_layernorm=SimpleNamespace(weight=rand(512)+1,variance_epsilon=1e-5),
        get_sin_cos=lambda positions,forward_batch=None:(cos,sin))
    obj._v3_scratch_cache=types.MethodType(prolog.NPUFusedMLAPreprocess._v3_scratch_cache,obj)
    monkeypatch.setattr(prolog,'get_token_to_kv_pool',lambda:pool)
    monkeypatch.setattr(prolog,'get_attn_backend',lambda:SimpleNamespace(use_flash_mla=True))
    monkeypatch.setattr(prolog,'is_mlaprolog_v3_direct_cache_enabled',lambda:direct_cache)
    batch=SimpleNamespace(out_cache_loc=idx)
    positions=torch.arange(tokens,device='npu:0')
    monkeypatch.setenv('SGLANG_NPU_MLAPO_V3_PACKED_QUERY','0')
    ref=prolog.NPUFusedMLAPreprocess.forward_mlaprolog_v3(obj,positions,hidden,batch)
    cache_ref=pool.kv_buffer.clone()
    expected=torch.cat([ref[2],ref[0]],-1)
    monkeypatch.setenv('SGLANG_NPU_MLAPO_V3_PACKED_QUERY','1')
    out=prolog.NPUFusedMLAPreprocess.forward_mlaprolog_v3(obj,positions,hidden,batch)
    torch.npu.synchronize()
    assert torch.equal(pool.kv_buffer,cache_ref)
    assert all(torch.equal(out[i],ref[i]) for i in [0,2,4])
    assert out[2].stride()==(20*576,576,1)
    # Capture the allocating high-level wrapper used by SGLang, including
    # both scratch/landing and direct-cache variants, not only the out API.
    graph=torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured=prolog.NPUFusedMLAPreprocess.forward_mlaprolog_v3(obj,positions,hidden,batch)
    graph.replay();torch.npu.synchronize()
    assert all(torch.equal(captured[i],out[i]) for i in [0,2,4])
    assert adapter.combine_query(captured[2],captured[0]).data_ptr()==captured[2].data_ptr()
    # Exercise the actual backend method and reject any concat on this path.
    meta=SimpleNamespace(block_tables=torch.zeros(tokens,1,device='npu:0',dtype=torch.int32),
        fmla_seqused_kv=torch.ones(tokens,device='npu:0',dtype=torch.int32),
        fmla_max_seqlen_kv=128)
    md=torch.empty(1,device='npu:0')
    backend=SimpleNamespace(forward_metadata=meta,graph_mode=True,kv_lora_rank=512,
        qk_rope_head_dim=64,_fmla_metadata_for=lambda **kw:md,token_to_kv_pool=pool)
    layer=SimpleNamespace(tp_q_head_num=20,layer_id=0,scaling=0.1)
    seen=[]
    def consume(query,*args,**kw):
        assert query.is_contiguous() and query.data_ptr()==out[2].data_ptr()
        assert torch.equal(query[:,0],expected)
        seen.append(True)
        return torch.zeros(tokens,1,20,512,device='npu:0',dtype=torch.bfloat16)
    def forbidden(*args,**kw):raise AssertionError('Query concat must not run')
    monkeypatch.setattr(adapter,'decode',consume)
    monkeypatch.setattr(torch,'cat',forbidden)
    result=AscendAttnBackend._fmla_decode(backend,out[2],out[0],layer,batch,tokens)
    assert seen and result.shape==(tokens,20*512)

@pytest.mark.parametrize('tokens',[1,7,64])
def test_real_flashmla(tokens):
    torch.manual_seed(42)
    full=torch.randn(tokens,1,20,576,device='npu:0',dtype=torch.bfloat16)
    query=adapter.combine_query(full[...,:512],full[...,512:])
    copied=torch.cat([full[...,:512],full[...,512:]],-1)
    cache=torch.randn(tokens,128,1,576,device='npu:0',dtype=torch.bfloat16)
    table=torch.arange(tokens,device='npu:0',dtype=torch.int32).view(tokens,1)
    lengths=torch.full((tokens,),127,device='npu:0',dtype=torch.int32)
    md=adapter.metadata(20,batch_size=tokens,max_seqlen_q=1,max_seqlen_kv=128,cache_seqlens=lengths)
    def run(q):return adapter.decode(q,cache,table,lengths,num_q_heads=20,scale=0.1,max_seqlen_kv=128,md=md)
    expected=run(copied);actual=run(query);torch.npu.synchronize()
    assert torch.equal(actual,expected)
