"""Direct latent-cache writes; Query layout and PD registration stay unchanged."""
from types import SimpleNamespace
import types

import pytest
import torch
import torch_npu
import sgl_kernel_npu
from sgl_kernel_npu.mla_prolog_v3_compat import direct

from sglang.srt.hardware_backend.npu.attention import mla_preprocess as prolog
from sglang.srt.hardware_backend.npu import memory_pool_npu as pools


pytestmark = pytest.mark.skipif(
    getattr(direct(), "mla_prolog_v3_direct_cache_version", 0) < 1,
    reason="requires sgl-kernel-npu with physical MLA cache-stride support",
)


def inputs(tokens):
    torch.manual_seed(42)
    def rand(*shape):
        return (torch.randn(*shape)*.02).to(dtype=torch.bfloat16,device='npu:0')
    def nz(*shape):
        return torch_npu.npu_format_cast(rand(*shape),29)
    return dict(token_x=rand(tokens,2048),weight_dq=nz(2048,768),
                weight_uq_qr=nz(768,20*256),weight_uk=rand(20,192,512),
                weight_dkv_kr=nz(2048,576),rmsnorm_gamma_cq=rand(768)+1,
                rmsnorm_gamma_ckv=rand(512)+1,rope_sin=rand(tokens,64),
                rope_cos=rand(tokens,64),query_norm_flag=True,
                weight_quant_mode=0,cache_mode='PA_BSND')


@pytest.mark.parametrize('page,tokens',[(16,1),(16,7),(16,17),(16,64),(128,1),(128,7),(128,64),(128,129)])
def test_direct_cache_matches_packed_and_preserves_other_slots(page,tokens):
    args=inputs(tokens)
    blocks=max(16,(tokens*3+page-1)//page)
    index=(torch.arange(tokens,device='npu:0',dtype=torch.int64)*17+page-1)%(blocks*page)
    assert index.unique().numel()==tokens
    k=torch.full((blocks,page,1,512),123,dtype=torch.bfloat16,device='npu:0')
    r=torch.full((blocks,page,1,64),123,dtype=torch.bfloat16,device='npu:0')
    expected=torch.ops.npu.mla_prolog_v3(**args,kv_cache=k,kr_cache=r,cache_index=index)
    full=torch.full((blocks,page,1,576),123,dtype=torch.bfloat16,device='npu:0')
    actual=torch.ops.npu.mla_prolog_v3(**args,kv_cache=full[...,:512],kr_cache=full[...,512:],cache_index=index)
    torch.npu.synchronize()
    assert torch.equal(full,torch.cat([k,r],dim=-1))
    for left,right in zip(expected,actual):
        assert torch.equal(left,right)
    # The second Cat must remain: prolog Query components are still separate.
    assert actual[0].is_contiguous() and actual[1].is_contiguous()
    assert actual[0].shape[-1]==512 and actual[1].shape[-1]==64


@pytest.mark.parametrize('tokens',[1,7,64])
def test_wrapper_skips_scratch_and_landing_keeps_pd_abi(monkeypatch,tokens):
    args=inputs(tokens)
    monkeypatch.setattr(pools,'use_unified_mla_kv_layout',lambda:True)
    pool=pools.NPUMLATokenToKVPool(size=2048,page_size=128,dtype=torch.bfloat16,
        kv_lora_rank=512,qk_rope_head_dim=64,layer_num=2,device='npu:0',
        enable_memory_saver=False,start_layer=4,end_layer=6)
    pool.kv_buffer.fill_(123)
    index=(torch.arange(tokens,device='npu:0',dtype=torch.int64)*17+127)%pool.size
    obj=SimpleNamespace(has_preprocess_weights=True,layer_id=5,kv_lora_rank=512,
        qk_rope_head_dim=64,num_local_heads=20,q_lora_rank=768,
        weight_dq=args['weight_dq'],weight_uq_qr=args['weight_uq_qr'],
        weight_uk=args['weight_uk'],weight_dkv_kr=args['weight_dkv_kr'],
        q_a_layernorm=SimpleNamespace(weight=args['rmsnorm_gamma_cq'],variance_epsilon=1e-5),
        kv_a_layernorm=SimpleNamespace(weight=args['rmsnorm_gamma_ckv'],variance_epsilon=1e-5),
        get_sin_cos=lambda positions:(args['rope_cos'],args['rope_sin']))
    obj._v3_scratch_cache=types.MethodType(prolog.NPUFusedMLAPreprocess._v3_scratch_cache,obj)
    monkeypatch.setattr(prolog,'get_token_to_kv_pool',lambda:pool)
    monkeypatch.setattr(prolog,'get_attn_backend',lambda:SimpleNamespace(use_flash_mla=True))
    monkeypatch.setattr(prolog,'is_mlaprolog_v3_direct_cache_enabled',lambda:False)
    batch=SimpleNamespace(out_cache_loc=index)
    positions=torch.arange(tokens,device='npu:0')
    expected=prolog.NPUFusedMLAPreprocess.forward_mlaprolog_v3(obj,positions,args['token_x'],batch)
    torch.npu.synchronize()
    expected_cache=pool.kv_buffer.clone()
    expected_q=[expected[i].clone() for i in [0,2,4]]
    descriptors=pool.get_contiguous_buf_infos()
    pool.kv_buffer.fill_(123)
    def forbidden(*args,**kwargs):raise AssertionError('scratch/landing must not run in direct mode')
    obj._v3_scratch_cache=forbidden
    monkeypatch.setattr(pool,'set_unified_mla_kv',forbidden)
    monkeypatch.setattr(prolog,'is_mlaprolog_v3_direct_cache_enabled',lambda:True)
    waits=[]
    pool.layer_transfer_counter=SimpleNamespace(wait_until=lambda layer:waits.append(layer))
    actual=prolog.NPUFusedMLAPreprocess.forward_mlaprolog_v3(obj,positions,args['token_x'],batch)
    torch.npu.synchronize()
    assert torch.equal(pool.kv_buffer,expected_cache)
    for i,reference in zip([0,2,4],expected_q):assert torch.equal(actual[i],reference)
    assert pool.get_contiguous_buf_infos()==descriptors
    assert waits and waits[0]==1


def test_direct_cache_graph_replay_uses_updated_slots():
    args=inputs(7)
    full=torch.full((4,128,1,576),123,dtype=torch.bfloat16,device='npu:0')
    index=torch.arange(7,device='npu:0',dtype=torch.int64)*17+127
    def run():
        return torch.ops.npu.mla_prolog_v3(**args,kv_cache=full[...,:512],kr_cache=full[...,512:],cache_index=index)
    stream=torch.npu.Stream()
    with torch.npu.stream(stream):
        for _ in range(2):run()
    stream.synchronize()
    graph=torch.npu.NPUGraph()
    with torch.npu.graph(graph,stream=stream):captured=run()
    for offset in [0,37,121]:
        slots=(torch.arange(7,device='npu:0',dtype=torch.int64)*17+offset)%512
        index.copy_(slots)
        full.fill_(123)
        graph.replay()
        torch.npu.synchronize()
        k=torch.full((4,128,1,512),123,dtype=torch.bfloat16,device='npu:0')
        r=torch.full((4,128,1,64),123,dtype=torch.bfloat16,device='npu:0')
        expected=torch.ops.npu.mla_prolog_v3(**args,kv_cache=k,kr_cache=r,cache_index=index)
        torch.npu.synchronize()
        assert torch.equal(full,torch.cat([k,r],dim=-1))
        assert torch.equal(captured[0],expected[0])
        assert torch.equal(captured[1],expected[1])
