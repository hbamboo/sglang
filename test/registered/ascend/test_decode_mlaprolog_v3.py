"""Decode scope, packed scratch addressing and latent PD ABI regression checks."""
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.hardware_backend.npu.attention import mla_preprocess as prolog
from sglang.srt.hardware_backend.npu import memory_pool_npu as pools


@pytest.mark.parametrize('enabled,v3,decode,expected',[
    (False,False,False,False),(False,True,True,False),
    (True,False,False,True),(True,False,True,True),
    (True,True,False,False),(True,True,True,True),
])
def test_scope_and_cache_handshake(monkeypatch,enabled,v3,decode,expected):
    monkeypatch.setattr(prolog,'is_mla_preprocess_enabled',lambda:enabled)
    monkeypatch.setattr(prolog,'is_mlaprolog_v3_enabled',lambda:v3)
    batch=SimpleNamespace(forward_mode=SimpleNamespace(is_decode=lambda:decode))
    assert prolog.is_mla_preprocess_for_batch(batch) is expected


@pytest.mark.parametrize('page,tokens',[(1,1),(16,17),(128,7),(128,129),(2048,64)])
def test_scratch_is_step_sized(page,tokens):
    obj=SimpleNamespace(kv_lora_rank=512,qk_rope_head_dim=64)
    pool=SimpleNamespace(store_dtype=torch.bfloat16,dtype=torch.bfloat16,page_size=page)
    hidden=torch.empty((tokens,2048),dtype=torch.bfloat16)
    kv,kr,index=prolog.NPUFusedMLAPreprocess._v3_scratch_cache(obj,hidden,pool)
    assert kv.is_contiguous() and kr.is_contiguous()
    assert kv.shape[1] in (16,128)
    assert kv.shape[0]*kv.shape[1] < tokens+kv.shape[1]
    assert kv.shape[-1]==512 and kr.shape[-1]==64
    assert index.tolist()==list(range(tokens))


def test_scratch_rejects_quantized_pool():
    obj=SimpleNamespace(kv_lora_rank=512,qk_rope_head_dim=64)
    pool=SimpleNamespace(store_dtype=torch.uint8,dtype=torch.bfloat16,page_size=128)
    with pytest.raises(NotImplementedError,match='unquantised'):
        prolog.NPUFusedMLAPreprocess._v3_scratch_cache(obj,torch.empty((1,2048)),pool)


@pytest.mark.parametrize('page,tokens',[(16,1),(16,7),(16,17),(128,1),(128,64),(128,129)])
def test_real_operator_scratch_and_unified_landing(monkeypatch,page,tokens):
    import sgl_kernel_npu  # register the installed custom operator
    import torch_npu

    torch.manual_seed(20261030+tokens)
    device='npu:0';dtype=torch.bfloat16
    def rand(*shape):
        return (torch.randn(*shape,dtype=torch.float32)*0.02).to(dtype).to(device)
    def nz(*shape):return torch_npu.npu_format_cast(rand(*shape),29)
    obj=SimpleNamespace(kv_lora_rank=512,qk_rope_head_dim=64)
    monkeypatch.setattr(pools,'use_unified_mla_kv_layout',lambda:True)
    pool=pools.NPUMLATokenToKVPool(size=8*page,page_size=page,dtype=dtype,
        kv_lora_rank=512,qk_rope_head_dim=64,layer_num=2,device=device,
        enable_memory_saver=False,start_layer=4,end_layer=6)
    x=rand(tokens,2048)
    scratch_k,scratch_r,scratch_index=prolog.NPUFusedMLAPreprocess._v3_scratch_cache(obj,x,pool)
    loc=(torch.arange(tokens,device=device,dtype=torch.int64)*3+page)%(pool.size)
    # tokens <=129 and size>=128*8 in the long case: these locations are unique.
    assert loc.unique().numel()==tokens
    packed_k=torch.zeros_like(pool.k_buffer[0]).contiguous()
    packed_r=torch.zeros_like(pool.v_buffer[0]).contiguous()
    shared=dict(token_x=x,weight_dq=nz(2048,768),weight_uq_qr=nz(768,20*256),
        weight_uk=rand(20,192,512),weight_dkv_kr=nz(2048,576),
        rmsnorm_gamma_cq=rand(768)+1,rmsnorm_gamma_ckv=rand(512)+1,
        rope_sin=rand(tokens,64),rope_cos=rand(tokens,64),
        rmsnorm_epsilon_cq=1e-5,rmsnorm_epsilon_ckv=1e-5,cache_mode='PA_BSND',
        query_norm_flag=True,weight_quant_mode=0)
    expected=torch.ops.npu.mla_prolog_v3(**shared,kv_cache=packed_k,kr_cache=packed_r,cache_index=loc)
    actual=torch.ops.npu.mla_prolog_v3(**shared,kv_cache=scratch_k,kr_cache=scratch_r,cache_index=scratch_index)
    torch.npu.synchronize()
    for index in [0,1,3]:assert torch.equal(expected[index],actual[index])
    latent=scratch_k.view(-1,512)[:tokens];rope=scratch_r.view(-1,64)[:tokens]
    assert torch.equal(latent,packed_k.view(-1,512)[loc])
    assert torch.equal(rope,packed_r.view(-1,64)[loc])
    before=pool.kv_buffer.clone()
    pool.set_unified_mla_kv(5,loc,latent,rope)
    reference=before.clone();reference[1].view(-1,576)[loc]=torch.cat([latent,rope],dim=-1)
    assert torch.equal(reference,pool.kv_buffer)
    ptrs,lengths,items=pool.get_contiguous_buf_infos()
    assert len(ptrs)==pool.layer_num==2
    assert items==[page*576*2]*2
    assert ptrs==[pool.kv_buffer[i].data_ptr() for i in range(2)]
    assert lengths==[pool.kv_buffer[i].nbytes for i in range(2)]


@pytest.mark.parametrize('tokens',[1,17,129,1024])
def test_real_operator_fused_prefill_sink_on_unified_views(tokens):
    """The torch_npu adapter must preserve sparse writes through strided views."""
    import torch_npu

    torch.manual_seed(20261030)
    page=128;blocks=max(2048,tokens*4)//page
    unified=torch.zeros(blocks,page,1,576,dtype=torch.bfloat16,device='npu:0')
    packed_k=torch.zeros(blocks,page,1,512,dtype=torch.bfloat16,device='npu:0')
    packed_r=torch.zeros(blocks,page,1,64,dtype=torch.bfloat16,device='npu:0')
    def rand(*shape):return torch.randn(*shape,dtype=torch.bfloat16,device='npu:0')
    x=rand(tokens,1,1,576);gamma=rand(512)
    cos=rand(tokens,1,1,64);sin=rand(tokens,1,1,64)
    loc=torch.arange(tokens,dtype=torch.int64,device='npu:0')*3+7
    kwargs=dict(k_rope_scale=None,c_kv_scale=None,k_rope_offset=None,c_kv_offset=None,
                epsilon=1e-5,cache_mode='PA_BNSD',is_output_kv=True)
    expected=torch_npu.npu_kv_rmsnorm_rope_cache(x,gamma,cos,sin,loc,packed_r,packed_k,**kwargs)
    actual=torch_npu.npu_kv_rmsnorm_rope_cache(x,gamma,cos,sin,loc,unified[...,512:],unified[...,:512],**kwargs)
    torch.npu.synchronize()
    assert torch.equal(expected[2],actual[2])
    assert torch.equal(expected[3],actual[3])
    assert torch.equal(unified,torch.cat([packed_k,packed_r],dim=-1))
