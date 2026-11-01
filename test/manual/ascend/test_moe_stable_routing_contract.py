"""Current repository helper plus actual graph config classes; metadata only.

No torch/torch_npu import, device query or library load. This verifies admission,
fallback/error behavior and unchanged native/quant call syntax, not device math.
"""
import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[3]
CORE=ROOT/'python/sglang/srt'


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m


class Spec:
    def __init__(self,shape,dtype,device='npu',contiguous=True):
        self.shape=shape;self.ndim=len(shape);self.dtype=dtype;self.device=SimpleNamespace(type=device);self.contiguous=contiguous
    def is_contiguous(self):return self.contiguous


class GateTests(unittest.TestCase):
    def setUp(self):
        self.torch=ModuleType('torch')
        self.torch.float8_e4m3fn='E4';self.torch.float8_e8m0fnu='E8';self.torch.int32='I32'
        self.torch.compiler=SimpleNamespace(is_compiling=lambda:False)
        self.torch.npu=SimpleNamespace(is_current_stream_capturing=lambda:False)
        runtime=ModuleType('sglang.srt.runtime_context');runtime.get_disagg=lambda:SimpleNamespace(disaggregation_mode='prefill')
        runtime.get_exec=lambda:self.exec
        self.modules=patch.dict(sys.modules,{'torch':self.torch,'sglang.srt.runtime_context':runtime})
        self.modules.start();self.addCleanup(self.modules.stop)
        self.config=load('_actual_stable_config',CORE/'model_executor/cuda_graph_config.py')
        cfg=self.config.CudaGraphConfig(decode=self.config.PhaseConfig(backend='full'),
            prefill=self.config.PhaseConfig(backend='disabled'))
        self.exec=SimpleNamespace(graph=SimpleNamespace(cuda_graph_config=cfg))
        self.helper=load('_actual_stable_helper',CORE/'hardware_backend/npu/moe/stable_routing.py')
        self.x,self.ids,self.scale=Spec((8192,2048),'E4'),Spec((8192,4),'I32'),Spec((8192,32,2),'E8')
    def eligible(self):return self.helper._eligible(self.x,self.ids,self.scale,64,4)
    def test_actual_phase_config_allows_prefill_disabled_while_decode_full(self):
        self.assertEqual(self.exec.graph.cuda_graph_config.decode.backend,'full')
        self.assertTrue(self.eligible())
    def test_nonprefill_none_and_each_graph_backend_fail_closed(self):
        with patch.object(self.helper,'get_disagg',return_value=SimpleNamespace(disaggregation_mode='decode')):
            self.assertFalse(self.eligible())
        cfg=self.exec.graph.cuda_graph_config
        for backend in ('full','breakable','tc_piecewise'):
            cfg.prefill.backend=backend;self.assertFalse(self.eligible())
        self.exec.graph.cuda_graph_config=None;self.assertFalse(self.eligible())
    def test_compile_and_current_stream_capture_are_excluded(self):
        self.torch.npu.is_current_stream_capturing=lambda:True;self.assertFalse(self.eligible())
        self.torch.compiler.is_compiling=lambda:True
        def forbidden():raise AssertionError('queried NPU capture while compiler active')
        self.torch.npu.is_current_stream_capturing=forbidden;self.assertFalse(self.eligible())
    def test_domain_rejected_before_optional_loader(self):
        with patch.object(self.helper,'_kernel_route',side_effect=AssertionError('optional import')):
            for experts,topk in ((32,4),(64,8)):
                self.assertIsNone(self.helper.try_stable_prequantized_routing(self.x,self.ids,self.scale,experts,topk))
            for value in (Spec((8191,2048),'E4'),Spec((36001,2048),'E4'),Spec((8192,4096),'E4'),Spec((8192,2048),'BF16')):
                self.assertIsNone(self.helper.try_stable_prequantized_routing(value,self.ids,self.scale,64,4))
    def test_optional_module_missing_is_fallback_and_inner_import_errors_propagate(self):
        for missing,expected_none in [('sgl_kernel_npu.moe_stable_routing',True),('other_dependency',False)]:
            def fake_import(name,*args,**kwargs):
                raise ModuleNotFoundError('missing',name=missing)
            self.helper._kernel_route.cache_clear()
            with patch('builtins.__import__',side_effect=fake_import):
                if expected_none:self.assertIsNone(self.helper._kernel_route())
                else:
                    with self.assertRaises(ModuleNotFoundError):self.helper._kernel_route()
    def test_kernel_error_not_hidden_and_success_logging_once(self):
        def fail(*args):raise RuntimeError('device launch failure')
        with patch.object(self.helper,'_kernel_route',return_value=fail):
            with self.assertRaisesRegex(RuntimeError,'device launch'):self.helper.try_stable_prequantized_routing(self.x,self.ids,self.scale,64,4)
        result=tuple(object() for _ in range(4))
        with patch.object(self.helper,'_kernel_route',return_value=lambda *a:result),patch.object(self.helper.logger,'info') as logger:
            for _ in range(2):self.assertIs(self.helper.try_stable_prequantized_routing(self.x,self.ids,self.scale,64,4),result)
            self.assertEqual(logger.call_count,1)
    def test_default_flag_off_and_explicit_opt_in(self):
        env=load('_actual_stable_env',CORE/'environ.py')
        with patch.dict(os.environ,{},clear=False):
            os.environ.pop('SGLANG_NPU_MOE_STABLE_ROUTING',None)
            self.assertFalse(env.envs.SGLANG_NPU_MOE_STABLE_ROUTING.get())
            os.environ['SGLANG_NPU_MOE_STABLE_ROUTING']='1'
            self.assertTrue(env.envs.SGLANG_NPU_MOE_STABLE_ROUTING.get())



if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(GateTests))
    raise SystemExit(not result.wasSuccessful())
