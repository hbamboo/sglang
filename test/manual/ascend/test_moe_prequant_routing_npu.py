"""Opt-in real-checkpoint GMM/finalizer, stream and graph correctness test.

No server is started. Requires a separately scheduled idle physical card.
Times are intentionally omitted: this can run while other cards load models.
"""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', required=True, type=int)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--checkpoint', type=Path,
                   default=Path('/mnt/sfs_turbo/models/chatdpo-glm-flash-0727-mxfp8'))
    p.add_argument('--layers', nargs='+', type=int, default=[1, 2])
    p.add_argument('--tokens', nargs='+', type=int, default=[8192, 17226])
    a = p.parse_args()
    import torch
    import torch_npu
    from safetensors import safe_open
    import sglang.srt.layers.quantization  # register before implementation imports
    from sglang.srt.hardware_backend.npu.moe import init_routing as routing
    from sglang.srt.hardware_backend.npu.moe.finalize_routing import NPUFinalizeRouting
    from sglang.srt.hardware_backend.npu.quantization.moe_methods import NPUMXFP8MoEMethod

    torch.set_num_threads(4)
    torch.manual_seed(1031)
    torch.npu.set_device(a.device)
    device = torch.device(f'npu:{a.device}')
    report = dict(status='running', device=str(device), layers=a.layers,
                  tokens=a.tokens, checkpoint=str(a.checkpoint), cases=[],
                  precision='exact bytes for routing/GMM1 quant/scale, exact BF16 GMM2/final output',
                  timing_claim='none; concurrent model loading may affect host submission')

    def save():
        a.output.parent.mkdir(parents=True, exist_ok=True)
        tmp = a.output.with_suffix('.tmp')
        tmp.write_text(json.dumps(report, indent=2))
        tmp.replace(a.output)

    @contextmanager
    def enabled(value):
        with patch.dict(os.environ, {'SGLANG_NPU_MOE_PREQUANT_ROUTING': '1' if value else '0'}), \
             patch.object(routing, 'get_disagg', return_value=SimpleNamespace(disaggregation_mode='prefill')):
            yield

    def exact_bytes(x, y):
        xb = x.detach().contiguous().view(torch.uint8).reshape(x.shape[0], -1).cpu()
        yb = y.detach().contiguous().view(torch.uint8).reshape(y.shape[0], -1).cpu()
        assert xb.shape == yb.shape and torch.equal(xb, yb), 'byte mismatch'

    index = json.loads((a.checkpoint / 'quant_model_weights.safetensors.index.json').read_text())['weight_map']
    init = routing.NPUMoEInitRouting_v2(quant_mode=3)
    finalize = NPUFinalizeRouting(drop_pad_mode=2)
    method13, method2 = NPUMXFP8MoEMethod('w13'), NPUMXFP8MoEMethod('w2')

    def load_layer(layer_index):
        # Read real offline-MXFP8 tensors; concatenate raw bytes to avoid a
        # CPU float8 arithmetic/cast or accidental re-quantization of weights.
        handles = {}

        def get(name):
            shard = index[name]
            if shard not in handles:
                handles[shard] = safe_open(a.checkpoint / shard, framework='pt', device='cpu')
            return handles[shard].get_tensor(name)

        w13, s13, w2, s2 = [], [], [], []
        for expert in range(64):
            prefix = f'model.layers.{layer_index}.mlp.experts.{expert}.'
            w13.append(torch.cat([get(prefix + part + '.weight').view(torch.uint8)
                                  for part in ('gate_proj', 'up_proj')], dim=0))
            s13.append(torch.cat([get(prefix + part + '.weight_scale').view(torch.uint8)
                                  for part in ('gate_proj', 'up_proj')], dim=0))
            w2.append(get(prefix + 'down_proj.weight').view(torch.uint8))
            s2.append(get(prefix + 'down_proj.weight_scale').view(torch.uint8))
        holder = torch.nn.Module()
        for name, values, dtype in [('w13_weight', w13, torch.float8_e4m3fn),
                                    ('w13_weight_scale', s13, torch.uint8),
                                    ('w2_weight', w2, torch.float8_e4m3fn),
                                    ('w2_weight_scale', s2, torch.uint8)]:
            tensor = torch.stack(values).view(dtype).to(device)
            setattr(holder, name, torch.nn.Parameter(tensor, requires_grad=False))
        method13.process_weights_after_loading(holder, 'w13')
        method2.process_weights_after_loading(holder, 'w2')
        return holder

    def expert_pipeline(holder, routed):
        payload, _, counts, scales = routed
        middle, middle_scales = method13.apply_fused_gmm1_swiglu(holder, payload, counts, scales, 1)
        output = method2.apply(holder, middle, counts, middle_scales, torch.bfloat16, 'w2', 1)
        return middle, middle_scales, output

    def run(holder, x, ids, weights, use_candidate):
        with enabled(use_candidate):
            routed = init._init_routing(x, ids, 64, 4)
        middle, middle_scales, output = expert_pipeline(holder, routed)
        final = finalize._finalize_routing(output, weights, routed[1], ids)
        return routed, middle, middle_scales, output, final

    save()
    try:
        for layer_index in a.layers:
            holder = load_layer(layer_index)
            report['loaded_weight_shapes'] = {name: list(value.shape)
                                               for name, value in holder.named_parameters()}
            for tokens in a.tokens:
                for distribution in ('uniform', 'skewed'):
                    x = torch.randn(tokens, 2048, device=device, dtype=torch.bfloat16)
                    if distribution == 'uniform':
                        ids = torch.rand(tokens, 64, device=device).topk(4, dim=-1).indices.to(torch.int32)
                    else:
                        ids = torch.arange(4, device=device, dtype=torch.int32).expand(tokens, 4).contiguous()
                    weights = torch.rand(tokens, 4, device=device).softmax(-1).to(torch.bfloat16)
                    assert weights.dtype == torch.bfloat16
                    expected = run(holder, x, ids, weights, False)
                    actual = run(holder, x, ids, weights, True)
                    torch.npu.synchronize()
                    for lhs, rhs in zip(expected[0], actual[0]):
                        exact_bytes(lhs, rhs)
                    exact_bytes(expected[1], actual[1])
                    exact_bytes(expected[2], actual[2])
                    torch.testing.assert_close(expected[3], actual[3], rtol=0, atol=0)
                    torch.testing.assert_close(expected[4], actual[4], rtol=0, atol=0)
                    # Independent expert-id oracle tests the finalizer/top-k
                    # weighting convention without deriving it from rowmap.
                    counts = actual[0][2].cpu().long()
                    expanded_experts = torch.repeat_interleave(torch.arange(64), counts).to(device)
                    synthetic = (expanded_experts.float() / 64).to(torch.bfloat16).unsqueeze(1).expand(-1, 32).contiguous()
                    synthetic_final = finalize._finalize_routing(synthetic, weights, actual[0][1], ids)
                    reference = ((ids.cpu().float() / 64) * weights.cpu().float()).sum(-1)
                    torch.testing.assert_close(synthetic_final.cpu().float(),
                                               reference[:, None].expand(-1, 32), rtol=0, atol=0.015625)
                    report['cases'].append(dict(layer=layer_index, tokens=tokens, distribution=distribution,
                                                route_bytes=True, gmm1_bytes=True, gmm2_exact=True,
                                                final_exact=True, topk_oracle_atol=0.015625))
                    save()
                    print(json.dumps(report['cases'][-1]), flush=True)
                    del expected, actual
            # Exercise current production wrapper on a nondefault caller and
            # captured graph with refreshed device values/route assignments.
            tokens = 8192
            x = torch.randn(tokens, 2048, device=device, dtype=torch.bfloat16)
            ids = torch.rand(tokens, 64, device=device).topk(4, dim=-1).indices.to(torch.int32)
            weights = torch.rand(tokens, 4, device=device).softmax(-1).to(torch.bfloat16)
            expected = run(holder, x, ids, weights, False)[-1]
            stream = torch.npu.Stream()
            stream.wait_stream(torch.npu.current_stream())
            with torch.npu.stream(stream):
                actual = run(holder, x, ids, weights, True)[-1]
                for _ in range(2):
                    run(holder, x, ids, weights, True)
            stream.synchronize()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                graph_output = run(holder, x, ids, weights, True)[-1]
            for factor in (0.25, -0.75):
                x.copy_(torch.randn_like(x) * factor)
                ids.copy_(torch.rand(tokens, 64, device=device).topk(4, dim=-1).indices.to(torch.int32))
                weights.copy_(torch.rand_like(weights.float()).softmax(-1).to(torch.bfloat16))
                expected = run(holder, x, ids, weights, False)[-1]
                graph.replay()
                torch.npu.synchronize()
                torch.testing.assert_close(graph_output, expected, rtol=0, atol=0)
            report['cases'].append(dict(layer=layer_index, nondefault_stream=True,
                                        graph_replays_with_new_values=2, final_exact=True))
            save()
            del holder, graph, graph_output, expected, actual
            torch.npu.synchronize()
        report['status'] = 'passed'
        save()
    except BaseException as exc:
        report['status'] = 'failed'
        report['error'] = f'{type(exc).__name__}: {exc}'
        save()
        raise


if __name__ == '__main__':
    main()
