"""Prune staged-overlap schedules with GLM dimensions and synthetic values.

This component fixture is NOT a substitute for customer TTFT or GSM8K.
It uses the same NPU kernels, tensor shapes and scaling as the model.
"""
import argparse
import json
import pathlib
import random
import runpy
import statistics
import time
from types import SimpleNamespace

import torch
import torch_npu
from sglang.srt.layers import quantization  # noqa: F401
from sglang.srt.hardware_backend.npu.moe.shared_expert_overlap import (
    SharedExpertOverlap,
    shared_expert_overlap_context,
)
from sglang.srt.hardware_backend.npu.quantization.moe_methods import NPUMXFP8MoEMethod
from sglang.srt.hardware_backend.npu.utils import npu_format_cast
from sglang.srt.layers.moe.moe_runner.ascend import (
    AscendQuantInfo,
    AscendRunnerCore,
    AscendRunnerInput,
)
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig

parser=argparse.ArgumentParser()
parser.add_argument('--output',required=True)
parser.add_argument('--tokens',default='1606,8192,30861,32768')
parser.add_argument('--repeat',type=int,default=15)
args=parser.parse_args()


@torch.inference_mode()
def main():
    torch.npu.set_device(0)
    torch.manual_seed(20261030)
    torch.npu.config.allow_internal_format=True
    fixture=runpy.run_path('/sgl-workspace/sglang/test/manual/ascend/test_glm_shared_expert_overlap.py',run_name='component_fixture')
    shared=fixture['Shared']()
    owner=SimpleNamespace(w13_kernel=NPUMXFP8MoEMethod('w13'),w2_kernel=NPUMXFP8MoEMethod('w2'))
    weights={}
    for prefix,n,k in [('w13',3072,2048),('w2',2048,1536)]:
        raw=torch.randn(64,n,k,device='npu',dtype=torch.bfloat16)*0.01
        q,scale=torch.ops.npu.npu_dynamic_mx_quant(raw,dst_type=torch.float8_e4m3fn)
        weights[prefix+'_weight']=npu_format_cast(q).transpose(1,2)
        weights[prefix+'_weight_scale']=scale.transpose(1,2)
        del raw,q,scale
    info=AscendQuantInfo(**weights)
    runner=AscendRunnerCore(MoeRunnerConfig(layer=owner,activation='silu'))
    gate_weight=torch.randn(64,2048,device='npu',dtype=torch.float32)*0.01
    bias=torch.zeros(64,device='npu',dtype=torch.float32)
    configs=[('serial',False,-1,-1),('staged',False,-1,-1),('side_c8_v16',False,8,16),('side_c16_v16',False,16,16),('side_c16_v32',False,16,32),('side_c24_v32',False,24,32),('side_c32_v16',False,32,16),('prequant',True,-1,-1),('prequant_c16_v16',True,16,16),('prequant_c24_v32',True,24,32),('prequant_c32_v16',True,32,16)]
    output=pathlib.Path(args.output)
    results=[]
    started=time.time()
    for tokens in map(int,args.tokens.split(',')):
        x=torch.randn(tokens,2048,device='npu',dtype=torch.bfloat16)

        def routed(logits):
            probabilities,indices,_=torch.ops.npu.npu_moe_gating_top_k(logits,k=4,bias=bias,k_group=1,group_count=1,group_select_mode=1,renorm=True,norm_type=1,routed_scaling_factor=1.0,eps=1e-20)
            probabilities=probabilities.to(x.dtype)
            q,permutation,counts,scales=torch.ops.npu.npu_moe_init_routing_v2(x,indices.to(torch.int32),active_num=tokens*4,expert_num=64,expert_tokens_num_type=1,expert_tokens_num_flag=True,active_expert_range=[0,64],quant_mode=3)
            scales=scales.reshape(q.shape[0],-1,2)
            y=runner.run(AscendRunnerInput(hidden_states=q,hidden_states_scale=scales,expert_tokens=counts.to(torch.int64),group_list_type=1),info,{}).hidden_states
            y=torch.ops.npu.npu_moe_finalize_routing(y,skip1=None,skip2=None,bias=None,scales=probabilities,expanded_src_to_dst_row=permutation,export_for_source_row=indices.to(torch.int32),drop_pad_mode=0)
            y*=1.8
            return y

        def serial():
            sh=shared(x)
            logits=torch.nn.functional.linear(x.float(),gate_weight)
            return routed(logits)+sh

        expected=serial()
        torch.npu.synchronize()
        for name,prequant,cubes,vectors in configs:
            overlap=None if name=='serial' else SharedExpertOverlap(owner,shared,prequant=prequant,cube_cores=cubes,vector_cores=vectors)

            def execute():
                if overlap is None:return serial()
                with shared_expert_overlap_context(overlap):
                    if prequant:overlap.prepare(x)
                    logits=torch.nn.functional.linear(x.float(),gate_weight)
                    overlap.start(x)
                    y=routed(logits)
                    sh=overlap.finish()
                    return y+sh

            torch.npu.empty_cache()
            for _ in range(5):actual=execute()
            torch.npu.synchronize()
            difference=float((actual.float()-expected.float()).abs().max())
            identical=torch.equal(actual,expected)
            assert identical,(tokens,name,difference)
            times=[]
            for _ in range(args.repeat):
                start,end=torch.npu.Event(enable_timing=True),torch.npu.Event(enable_timing=True)
                start.record();actual=execute();end.record();end.synchronize();times.append(start.elapsed_time(end))
            record=dict(tokens=tokens,variant=name,prequant=prequant,cube_cores=cubes,vector_cores=vectors,bitwise_equal=identical,max_abs=difference,median_ms=statistics.median(times),mean_ms=statistics.mean(times),times_ms=times)
            results.append(record)
            output.write_text(json.dumps(dict(fixture='synthetic GLM dimensions',seconds=time.time()-started,results=results),indent=2))
            print(json.dumps({k:v for k,v in record.items() if k!='times_ms'}),flush=True)
        del x,expected,actual
    print('COMPLETE',len(results),'cases',flush=True)


if __name__=='__main__':main()
