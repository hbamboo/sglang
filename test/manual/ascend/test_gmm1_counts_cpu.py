"""CPU call-contract tests; no Torch/NPU import and no numerical claim."""
import ast
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'python/sglang/srt/hardware_backend/npu/moe/matmul.py'
ENVIRON = ROOT / 'python/sglang/srt/environ.py'


class Tensor:
    def __init__(self, shape, dtype, device, contiguous=True):
        self.shape, self.dtype, self.device = tuple(shape), dtype, device
        self.ndim, self.contiguous = len(shape), contiguous
        self.cumsum_calls, self.cumulative = 0, object()

    def is_contiguous(self):
        return self.contiguous

    def cumsum(self, dim):
        assert dim == 0
        self.cumsum_calls += 1
        return self.cumulative


class Packet:
    def __init__(self):
        self.calls, self.error = [], None
        self.output = (object(), object())

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.output


class TestGMM1CountsContract(unittest.TestCase):
    def setUp(self):
        self.flag, self.compiling, self.capturing = False, False, False
        self.mode, self.world, self.backend, self.tc = 'prefill', 1, 'disabled', None
        self.flag_reads, self.runtime_reads, self.capture_reads = 0, 0, 0
        self.packet = Packet()
        self.device = SimpleNamespace(type='npu', index=0)
        self.torch = ModuleType('torch')
        self.torch.Tensor, self.torch.dtype = Tensor, object
        self.torch.nn = SimpleNamespace(Module=object)
        self.torch.float8_e4m3fn, self.torch.float32, self.torch.int64 = 'e4m3', 'fp32', 'i64'
        self.torch.compiler = SimpleNamespace(is_compiling=lambda: self.compiling)
        self.torch.npu = SimpleNamespace(is_current_stream_capturing=self.capture)
        self.torch.ops = SimpleNamespace(npu=SimpleNamespace(npu_grouped_matmul_swiglu_quant_v2=self.packet))
        env = ModuleType('sglang.srt.environ')
        env.envs = SimpleNamespace(SGLANG_NPU_GMM1_COUNTS=SimpleNamespace(get=self.flag_value))
        runtime = ModuleType('sglang.srt.runtime_context')
        runtime.get_disagg = lambda: self.runtime_value(SimpleNamespace(disaggregation_mode=self.mode))
        runtime.get_parallel = lambda: self.runtime_value(SimpleNamespace(world_size=self.world))
        runtime.get_exec = lambda: self.runtime_value(SimpleNamespace(graph=SimpleNamespace(
            cuda_graph_config=SimpleNamespace(prefill=SimpleNamespace(backend=self.backend)))))
        tc = ModuleType('sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph')
        tc.get_tc_piecewise_forward_context = lambda: self.tc
        self.modules = patch.dict(sys.modules, {'torch': self.torch, env.__name__: env,
                                               runtime.__name__: runtime, tc.__name__: tc})
        self.modules.start()
        self.module_name = '_counts_cpu_contract_' + str(id(self))
        spec = importlib.util.spec_from_file_location(self.module_name, SOURCE)
        self.module = importlib.util.module_from_spec(spec)
        sys.modules[self.module_name] = self.module
        spec.loader.exec_module(self.module)
        self.method = self.module.GroupedMatmulSwigluQuant()

    def tearDown(self):
        sys.modules.pop(self.module_name, None)
        self.modules.stop()

    def flag_value(self):
        self.flag_reads += 1
        return self.flag

    def runtime_value(self, value):
        self.runtime_reads += 1
        return value

    def capture(self):
        self.capture_reads += 1
        return self.capturing

    def inputs(self, m=32768):
        x = Tensor((m, 2048), 'e4m3', self.device)
        counts = Tensor((64,), 'i64', self.device)
        weight = Tensor((64, 2048, 3072), 'e4m3', self.device, contiguous=False)
        holder = SimpleNamespace(w13_weight=weight)
        scales = dict(dequant_mode=2, quant_mode=2, dequant_dtype='fp32', quant_dtype='e4m3',
                      weight_scale_dtype='e8m0-enum', x_scale_dtype='e8m0-enum',
                      weight_scale=[object()], x_scale=object())
        return holder, x, counts, scales

    def forward(self, values, group_list_type=1):
        holder, x, counts, scales = values
        return self.method.forward(holder, 'w13', x, counts,
                                   group_list_type=group_list_type, transposed=True, **scales)

    def assert_legacy(self, values, group_list_type=1):
        holder, x, counts, scales = values
        previous = counts.cumsum_calls
        output = self.forward(values, group_list_type)
        self.assertIs(output, self.packet.output)
        kwargs = self.packet.calls[-1]
        self.assertEqual(kwargs, dict(x=x, weight=[holder.w13_weight],
                                     group_list=counts.cumulative if group_list_type == 1 else counts, **scales))
        self.assertEqual(counts.cumsum_calls - previous, int(group_list_type == 1))
        self.assertNotIn('group_list_type', kwargs)  # original API's default0

    def test_new_flag_declared_default_false(self):
        tree = ast.parse(ENVIRON.read_text(encoding='utf-8'))
        entry = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == 'SGLANG_NPU_GMM1_COUNTS' for t in n.targets))
        self.assertIs(entry.value.args[0].value, False)

    def test_flag0_original_call_and_no_runtime_or_capture_query(self):
        self.assert_legacy(self.inputs())
        self.assertEqual((self.runtime_reads, self.capture_reads), (0, 0))
        self.assertIs(self.torch.ops.npu.npu_grouped_matmul_swiglu_quant_v2, self.packet)

    def test_direct_counts_preserves_tensor_args_and_return(self):
        self.flag = True
        holder, x, counts, scales = values = self.inputs()
        self.assertIs(self.forward(values), self.packet.output)
        self.assertEqual(counts.cumsum_calls, 0)
        self.assertEqual(self.packet.calls[-1], dict(x=x, weight=[holder.w13_weight],
                                                   group_list=counts, group_list_type=1, **scales))
        self.assertEqual(self.capture_reads, 1)

    def test_small_odd_and_upper_domain(self):
        self.flag = True
        for m in (4, 512, 8192, 32772, 65536):
            with self.subTest(expanded_m=m):
                values = self.inputs(m)
                self.forward(values)
                self.assertIs(self.packet.calls[-1]['group_list'], values[2])
                self.assertEqual(values[2].cumsum_calls, 0)
        for m in (0, 3, 65540):
            with self.subTest(excluded_m=m):
                self.assert_legacy(self.inputs(m))

    def test_other_dtype_shape_or_quant_mode_falls_back(self):
        self.flag = True
        mutations = [lambda v: setattr(v[0].w13_weight, 'shape', (64, 2048, 2048)),
                     lambda v: setattr(v[0].w13_weight, 'dtype', 'fp4'),
                     lambda v: setattr(v[1], 'dtype', 'bf16'),
                     lambda v: setattr(v[1], 'contiguous', False),
                     lambda v: setattr(v[2], 'dtype', 'i32'),
                     lambda v: setattr(v[2], 'shape', (65,)),
                     lambda v: setattr(v[2], 'device', SimpleNamespace(type='cpu')),
                     lambda v: v[3].update(quant_mode=0),
                     lambda v: v[3].update(dequant_mode=0),
                     lambda v: v[3].update(dequant_dtype='bf16'),
                     lambda v: v[3].update(weight_scale_dtype=None),
                     lambda v: v[3].update(x_scale_dtype='different')]
        for index, mutate in enumerate(mutations):
            with self.subTest(case=index):
                values = self.inputs()
                mutate(values)
                self.assert_legacy(values)
        self.assertEqual(self.capture_reads, 0)

    def test_existing_cumulative_input_is_unchanged(self):
        self.flag = True
        self.assert_legacy(self.inputs(), group_list_type=0)
        self.assertEqual(self.capture_reads, 0)

    def test_decode_multirank_graph_tc_and_capture_fall_back(self):
        self.flag = True
        for key, value in [('mode', 'decode'), ('world', 2), ('backend', 'full'),
                           ('tc', object()), ('capturing', True)]:
            with self.subTest(field=key):
                original = getattr(self, key)
                setattr(self, key, value)
                self.assert_legacy(self.inputs())
                setattr(self, key, original)

    def test_compile_short_circuits_before_environment_and_context(self):
        self.flag = self.compiling = True
        self.assert_legacy(self.inputs())
        self.assertEqual((self.flag_reads, self.runtime_reads, self.capture_reads), (0, 0, 0))

    def test_device_api_error_propagates_without_retry_or_cumsum(self):
        self.flag = True
        self.packet.error = RuntimeError('synthetic API failure')
        values = self.inputs()
        with self.assertRaisesRegex(RuntimeError, 'synthetic API failure'):
            self.forward(values)
        self.assertEqual((len(self.packet.calls), values[2].cumsum_calls), (1, 0))
        self.assertFalse(self.module._logged_mxfp8_counts)

    def test_first_success_log_once(self):
        self.flag = True
        with patch.object(self.module.logger, 'info') as log:
            self.forward(self.inputs())
            self.forward(self.inputs())
        self.assertEqual(log.call_count, 1)


if __name__ == '__main__':
    unittest.main()
