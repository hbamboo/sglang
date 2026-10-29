"""Run on one Ascend NPU: ASCEND_RT_VISIBLE_DEVICES=5 python <this file>."""
import json
from types import SimpleNamespace

import torch
import torch_npu  # noqa: F401
from torch import nn

# Initialize the registry in the same order as the model loader; importing a
# concrete NPU quant method before this registry exposes an existing cycle.
from sglang.srt.layers import quantization  # noqa: F401
from sglang.srt.hardware_backend.npu.moe.shared_expert_overlap import (
    SharedExpertOverlap,
    can_overlap_shared_expert,
    get_shared_expert_overlap,
    shared_expert_overlap_context,
)
from sglang.srt.hardware_backend.npu.quantization.linear_method_npu import (
    NPUMXFP8LinearMethod,
)
from sglang.srt.layers.activation import SiluAndMul


class Projection(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(output_size, input_size, device='npu', dtype=torch.bfloat16) * 0.01,
            requires_grad=False,
        )
        self.bias = None
        self.tp_size = 1
        self.quant_method = NPUMXFP8LinearMethod()
        self.quant_method.process_weights_after_loading(self)

    def forward(self, x):
        return self.quant_method.apply(self, x), None


class Shared(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_up_proj = Projection(2048, 3072)
        self.down_proj = Projection(1536, 2048)
        self.act_fn = SiluAndMul()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_up_proj(x)[0]))[0]


@torch.inference_mode()
def main():
    torch.npu.set_device(0)
    torch.manual_seed(20261030)
    shared = Shared()
    owner = object()
    overlap = SharedExpertOverlap(owner, shared)
    parent = torch.npu.Stream()
    parent.wait_stream(torch.npu.current_stream())
    records = []
    with torch.npu.stream(parent):
        for tokens in [1, 128, 1024, 32768, 1024, 1024, 1024]:
            x = torch.randn(tokens, 2048, device='npu', dtype=torch.bfloat16)
            expected = shared(x)
            with shared_expert_overlap_context(overlap):
                assert get_shared_expert_overlap(owner) is overlap
                assert get_shared_expert_overlap(object()) is None
                overlap.start(x)
                # Independent real device work around the runner boundaries.
                gate_scores = x.float().sum(-1)
                routed1 = x * 0.5
                overlap.before_gmm2()
                routed2 = routed1 + 1
                overlap.after_gmm2()
                finalized = routed2.sum(-1) + gate_scores
                actual = overlap.finish()
            assert get_shared_expert_overlap(owner) is None
            del x, routed1, routed2, finalized, gate_scores
            # Allocate on the caller's non-default stream before observing
            # results, exercising recorded input/output allocation lifetimes.
            churn = torch.empty(tokens, 2048, device='npu', dtype=torch.bfloat16)
            churn.fill_(13)
            equal = torch.equal(actual, expected)
            maximum = float((actual.float() - expected.float()).abs().max())
            assert equal, (tokens, maximum)
            records.append(dict(tokens=tokens, bitwise_equal=equal, max_abs=maximum))
        try:
            with shared_expert_overlap_context(overlap):
                overlap.start(torch.ones(128, 2048, device='npu', dtype=torch.bfloat16))
                raise ValueError('exercise exceptional cleanup')
        except ValueError:
            pass
        assert get_shared_expert_overlap(owner) is None and overlap.main is None
        # Decode is deliberately excluded before touching backend attributes.
        block = SimpleNamespace(shared_experts=shared, experts=owner)
        forward = SimpleNamespace(forward_mode=SimpleNamespace(is_extend_without_speculative=lambda: False))
        assert not can_overlap_shared_expert(block, churn, forward)
    torch.npu.synchronize()
    print(json.dumps(dict(status='passed', logical_device=0, cases=records, exception_cleanup=True, decode_fallback=True), indent=2))


if __name__ == '__main__':
    main()
